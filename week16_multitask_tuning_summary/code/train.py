import argparse
import json
import os
import random
from datetime import datetime

import torch
import yaml
# Accelerator 负责把普通 PyTorch 训练脚本扩展到单卡、多卡、混合精度等场景。
from accelerate import Accelerator
# accelerate_set_seed 会额外处理分布式训练中的随机种子同步。
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
# 复用 transformers 提供的常见学习率调度器，避免手写 warmup/cosine 逻辑。
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# 数据集负责读取图片和问答文本，Collator 负责在 batch 阶段 tokenize 并构造 labels。
from dataset import LlavaCollator, LlavaPretrainDataset, MultiTaskLlavaDataset
# MiniLlavaModel 封装视觉编码器、投影层和语言模型解码器。
from mini_llava import MiniLlavaModel


def append_jsonl(path, record):
    """向 JSONL 日志追加一条结构化记录。"""
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_seed(seed):
    """固定随机种子，方便复现实验结果。"""
    # Python 自带 random 模块的随机性。
    random.seed(seed)
    # CPU 上的 PyTorch 随机性。
    torch.manual_seed(seed)
    # 所有 CUDA 设备上的 PyTorch 随机性；没有 GPU 时调用也不会影响训练。
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    """读取训练配置。"""
    # 配置文件使用 YAML，训练参数、数据路径、模型路径都从这里读取。
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_optimizer(model, config):
    """根据配置创建优化器，只更新 requires_grad=True 的参数。"""
    optim_config = config["TRAINING"]["OPTIMIZER"]
    base_lr = float(optim_config["LR"])
    module_configs = {
        "vision_encoder": config["MINILLAVA"].get("VISION_ENCODER", {}),
        "language_decoder": config["MINILLAVA"].get("LLM_DECODER", {}),
        "projector": config["MINILLAVA"].get("PROJECTOR", {}),
    }

    param_groups = []
    for module_name, module_config in module_configs.items():
        module = getattr(model, module_name)
        trainable_params = [p for p in module.parameters() if p.requires_grad]
        if not trainable_params:
            continue
        lr = module_config.get("LR")
        if lr is None:
            lr = base_lr
        param_groups.append({
            "params": trainable_params,
            "lr": float(lr),
            "name": module_name,
        })

    optim_type = optim_config["TYPE"].lower()
    # AdamW 是大语言模型微调中最常用的优化器，带 decoupled weight decay。
    if optim_type == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=float(optim_config["WEIGHT_DECAY"]),
            betas=tuple(optim_config["BETAS"])
        )
    # Adam 不使用 AdamW 的解耦权重衰减，适合简单调试或对比实验。
    if optim_type == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr)
    # SGD 一般不用于 LLM 微调，但保留入口便于实验。
    if optim_type == "sgd":
        return torch.optim.SGD(param_groups, lr=base_lr, momentum=0.9)
    raise ValueError(f"不支持的优化器类型: {optim_type}")


def as_list(value):
    """把单值配置统一成列表，兼容旧版单数据集 YAML。"""
    if isinstance(value, list):
        return value
    return [value]


def build_train_dataset(train_config, max_samples=None, split="train", split_seed=42):
    """根据 DATA.TRAIN_DATASET 构造单任务或多任务数据集。"""
    paths = as_list(train_config["PATH"])
    image_dirs = as_list(train_config["IMAGE_DIR"])
    annotation_files = as_list(train_config["ANNOTATION_FILE"])
    sample_rates = train_config.get("SAMPLE_RATE")
    sample_rates = as_list(sample_rates) if sample_rates is not None else None
    train_split = float(train_config.get("TRAIN_SPLIT", 1.0))

    dataset_count = len(paths)
    if not (len(image_dirs) == len(annotation_files) == dataset_count):
        raise ValueError("PATH、IMAGE_DIR、ANNOTATION_FILE 的数量必须一致。")
    if sample_rates is not None and len(sample_rates) != dataset_count:
        raise ValueError("SAMPLE_RATE 的数量必须和数据集数量一致。")

    datasets = []
    for idx, (dataset_path, image_dir, annotation_file) in enumerate(zip(paths, image_dirs, annotation_files)):
        task_name = train_config.get("TASK_NAME")
        if isinstance(task_name, list):
            current_task_name = task_name[idx]
        elif task_name:
            current_task_name = str(task_name)
        else:
            current_task_name = f"task_{idx}:{dataset_path}"

        datasets.append(
            LlavaPretrainDataset(
                dataset_path=dataset_path,
                image_dir=image_dir,
                annotation_file=annotation_file,
                max_samples=max_samples,
                task_name=current_task_name,
                split=split,
                train_split=train_split,
                split_seed=int(split_seed) + idx
            )
        )

    if dataset_count == 1:
        return datasets[0]

    return MultiTaskLlavaDataset(
        datasets=datasets,
        task_names=[dataset.task_name for dataset in datasets],
        sample_rates=sample_rates if split == "train" else None
    )


def build_train_sampler(dataset):
    """多任务数据集按 SAMPLE_RATE 构造采样器；单任务保持 DataLoader shuffle。"""
    if not isinstance(dataset, MultiTaskLlavaDataset):
        return None

    sample_weights = dataset.build_sample_weights()
    if sample_weights is None:
        return None

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )


def build_scheduler(optimizer, config, total_steps):
    """创建学习率调度器。"""
    # 调度器配置决定 warmup 步数、总训练步数下学习率如何变化。
    sched_config = config["TRAINING"]["SCHEDULER"]
    warmup_steps = int(sched_config["WARMUP_STEPS"])
    sched_type = sched_config["TYPE"].lower()
    # cosine：warmup 后按余弦曲线逐渐衰减，常用于 Transformer 训练。
    if sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # linear：warmup 后线性衰减到 0，行为更直观。
    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # constant：不使用 scheduler，训练过程中学习率保持 optimizer 初始值。
    if sched_type == "constant":
        return None
    raise ValueError(f"不支持的调度器类型: {sched_type}")


def should_save_param(name, param):
    """checkpoint 中保留 projector、可训练参数和 LoRA adapter 参数。"""
    name = name.lower()
    return name.startswith("projector.") or param.requires_grad or "lora_" in name or ".lora_" in name


def load_checkpoint(model, checkpoint_path):
    """从已有 checkpoint 初始化模型参数，不恢复优化器状态。"""
    if checkpoint_path is None:
        return None

    state = torch.load(checkpoint_path, map_location=model.device)
    state_dict = state.get("model", state)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "path": checkpoint_path,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "loaded_keys": list(state_dict.keys()),
    }


def save_checkpoint(accelerator, model, optimizer, scheduler, step, save_dir, filename="last.pt", val_loss=None):
    """保存训练检查点，包含 projector、可训练参数和优化器状态。"""
    # 多卡训练时每个进程都会执行代码；只让主进程写文件，避免多个进程同时覆盖同一路径。
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, filename)
    # prepare 后的 model 可能被 DDP/FSDP 等包装；保存前要取回原始模型对象。
    unwrapped_model = accelerator.unwrap_model(model)
    # 保存 projector、当前可训练参数和 LoRA adapter；冻结的基础 LLM/vision 权重由初始化模型提供。
    saved_param_names = [
        name
        for name, param in unwrapped_model.named_parameters()
        if should_save_param(name, param)
    ]
    saved_state_dict = {
        name: param.detach().cpu()
        for name, param in unwrapped_model.named_parameters()
        if should_save_param(name, param)
    }
    # accelerator.save 会在分布式环境中安全保存对象，语义类似 torch.save。
    accelerator.save(
        {
            # 记录当前全局步数，方便后续恢复或排查 checkpoint 来源。
            "step": step,
            "val_loss": val_loss,
            # 只保存部分参数；加载时用参数名称匹配，并允许未保存的冻结参数缺失。
            "model": saved_state_dict,
            # 显式记录保存了哪些参数，便于检查 checkpoint 内容。
            "saved_param_names": saved_param_names,
            # 保存优化器状态，恢复训练时可以延续动量等内部统计量。
            "optimizer": optimizer.state_dict(),
            # constant scheduler 为 None，其它 scheduler 保存状态用于恢复学习率进度。
            "scheduler": scheduler.state_dict() if scheduler is not None else None
        },
        ckpt_path
    )
    print(f"已保存检查点: {ckpt_path}")


def evaluate_validation_loss(accelerator, model, val_dataloader, step=None):
    """在验证集上计算平均 loss。"""
    if val_dataloader is None:
        if accelerator.is_main_process:
            print("未配置验证集，跳过 validation。")
        return None

    if accelerator.is_main_process:
        step_text = f" step={step}" if step is not None else ""
        print(f"开始验证{step_text}: batches={len(val_dataloader)}")

    model.eval()
    total_loss = 0.0
    total_count = 0.0
    val_iter = tqdm(
        val_dataloader,
        desc="validation",
        disable=not accelerator.is_main_process,
        leave=False
    )
    with torch.no_grad():
        for batch in val_iter:
            outputs = model(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            batch_size = batch["input_ids"].size(0)
            loss_sum = outputs.loss.detach() * batch_size
            loss_tensor = torch.tensor([loss_sum.item(), float(batch_size)], device=accelerator.device)
            gathered = accelerator.gather_for_metrics(loss_tensor)
            total_loss += gathered[0::2].sum().item()
            total_count += gathered[1::2].sum().item()
            if accelerator.is_main_process and total_count > 0:
                val_iter.set_postfix(loss=f"{total_loss / total_count:.4f}")

    model.train()
    if total_count == 0:
        if accelerator.is_main_process:
            print("验证集为空，无法计算 validation loss。")
        return None

    val_loss = total_loss / total_count
    if accelerator.is_main_process:
        print(f"验证完成: loss={val_loss:.4f}, samples={int(total_count)}")
    return val_loss


def main():
    # 命令行参数只保留配置路径和调试样本数，其他训练参数统一放在 YAML 中管理。
    parser = argparse.ArgumentParser(description="MiniLLaVA 微调脚本")
    parser.add_argument("--config", default="week14_dialogue_stability_output_control/configs/config.yaml", help="训练配置文件路径")
    parser.add_argument("--max-samples", type=int, default=None, help="调试时只取前 N 条数据")
    args = parser.parse_args()

    # Accelerator 会根据 accelerate launch 的启动方式自动识别进程数、设备和分布式后端。
    accelerator = Accelerator()
    config = load_config(args.config)
    # 保留原本的 PyTorch 随机种子设置。
    set_seed(int(config["MISC"]["SEED"]))
    # 再使用 Accelerate 的种子工具，保证多进程场景下每个进程的随机状态可控。
    accelerate_set_seed(int(config["MISC"]["SEED"]))

    # 初始化 MiniLLaVA，并按需从上一阶段 checkpoint 加载 projector/LoRA 等增量参数。
    model = MiniLlavaModel(args.config)
    load_path = config["TRAINING"].get("CHECKPOINT", {}).get("LOAD_PATH")
    load_info = load_checkpoint(model, load_path)
    if load_info is not None and accelerator.is_main_process:
        print(f"已加载检查点: {load_info['path']} loaded={len(load_info['loaded_keys'])} missing={len(load_info['missing_keys'])} unexpected={len(load_info['unexpected_keys'])}")
    model.train()

    # 从配置中读取训练数据路径、图片目录、标注文件、采样比例、batch size 等数据相关参数。
    train_config = config["DATA"]["TRAIN_DATASET"]
    dataset = build_train_dataset(train_config, max_samples=args.max_samples, split="train", split_seed=int(config["MISC"]["SEED"]))
    train_split = float(train_config.get("TRAIN_SPLIT", 1.0))
    val_dataset = build_train_dataset(train_config, max_samples=args.max_samples, split="val", split_seed=int(config["MISC"]["SEED"])) if train_split < 1 else None
    if val_dataset is not None and len(val_dataset) == 0:
        val_dataset = None
    sampler = build_train_sampler(dataset)
    if accelerator.is_main_process:
        if isinstance(dataset, MultiTaskLlavaDataset):
            if dataset.sample_rates is None:
                task_desc = ", ".join(
                    f"{name}: n={len(task_dataset)}"
                    for name, task_dataset in zip(dataset.task_names, dataset.datasets)
                )
            else:
                task_desc = ", ".join(
                    f"{name}: n={len(task_dataset)}, rate={rate}"
                    for name, task_dataset, rate in zip(dataset.task_names, dataset.datasets, dataset.sample_rates)
                )
            print(f"启用多任务混合训练: {task_desc}")
        else:
            print(f"启用单任务训练: n={len(dataset)}")
    # Collator 在 DataLoader 拼 batch 时进行 tokenizer、padding，并构造只监督 answer 的 labels。
    collator = LlavaCollator(
        tokenizer=model.language_decoder.tokenizer,
        max_length=int(train_config["MAX_LENGTH"]),
        max_turns=int(train_config.get("MAX_TURNS", 4)),
    )
    # DataLoader 仍按普通 PyTorch 写法创建；后面 accelerator.prepare 会自动处理多卡分片。
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["BATCH_SIZE"]),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(train_config["NUM_WORKERS"]),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(train_config["BATCH_SIZE"]),
            shuffle=False,
            num_workers=int(train_config["NUM_WORKERS"]),
            collate_fn=collator,
            pin_memory=torch.cuda.is_available()
        )

    # 先在原始 model 上创建优化器，这样 optimizer 能拿到正确的可训练参数列表。
    optimizer = build_optimizer(model, config)
    num_epochs = int(config["TRAINING"]["SCHEDULER"]["NUM_EPOCHS"])
    # prepare 会把 model 放到正确设备，并在多卡时包装为分布式模型；
    # dataloader 也会被切成每个进程各自负责的一份数据。
    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # prepare 之后的 dataloader 长度是当前进程实际迭代步数，用它计算 scheduler 总步数更适合多卡。
    total_steps = len(dataloader) * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)
    # scheduler 依赖已经 prepare 过的 optimizer，因此在 optimizer prepare 之后创建并交给 accelerator。
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # 日志、保存、梯度裁剪等训练控制参数。
    log_steps = int(config["TRAINING"]["LOGGING"]["LOG_STEPS"])
    log_dir = config["TRAINING"]["LOGGING"].get("LOG_DIR", "week14_dialogue_stability_output_control/outputs/logs")
    train_log_path = os.path.join(log_dir, "train.jsonl")
    save_steps = int(config["TRAINING"]["CHECKPOINT"]["SAVE_STEPS"])
    save_dir = config["TRAINING"]["CHECKPOINT"]["SAVE_DIR"]
    max_norm = float(config["TRAINING"]["GRAD_CLIP"]["MAX_NORM"])
    if accelerator.is_main_process:
        append_jsonl(train_log_path, {
            "event": "train_start",
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "num_epochs": num_epochs,
            "total_steps": total_steps,
            "max_samples": args.max_samples,
            "load_checkpoint": load_path,
            "train_split": train_split,
            "val_samples": len(val_dataset) if val_dataset is not None else 0,
        })

    # global_step 记录当前进程执行的优化步数；多卡下各进程同步前进。
    best_val_loss = None
    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            # batch 来自 LlavaCollator：
            # images 是 PIL 图片列表，input_ids/attention_mask/labels 是已 padding 的张量。
            outputs = model(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            # MiniLlavaModel 最终调用语言模型，labels 存在时 transformers 输出中会包含 loss。
            loss = outputs.loss
            # 使用 Accelerator 进行反向传播，兼容多卡、混合精度和梯度累积等能力。
            accelerator.backward(loss)

            # 梯度裁剪可以缓解训练初期或小 batch 时的梯度爆炸。
            accelerator.clip_grad_norm_(model.parameters(), max_norm)
            # 参数更新。
            optimizer.step()
            # 如果启用了 scheduler，每个优化步后推进一次学习率。
            if scheduler is not None:
                scheduler.step()
            # set_to_none=True 可以减少显存写入，下一次 backward 时再重新分配梯度。
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            # 聚合所有进程上的 loss，日志展示的是多卡平均 loss，而不是单个进程的局部 loss。
            loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
            non_padded_tokens = (batch["labels"] != -100).sum().item()
            total_tokens = batch["labels"].shape[0] * batch["labels"].shape[1]
            
            # 只让全局主进程打印日志，避免多进程重复输出相同 step。
            if global_step % log_steps == 0 and accelerator.is_main_process:
                lrs = {group.get("name", f"group_{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
                lr_text = ", ".join(f"{name}={lr:.8f}" for name, lr in lrs.items())
                print(f"epoch {epoch + 1}/{num_epochs} step={global_step} loss={loss_value:.4f} lr={lr_text} non-padded tokens={non_padded_tokens / total_tokens * 100:.2f}%")
                # print(f"Number of non-padded tokens: {non_padded_tokens / total_tokens * 100:.2f}%")
                append_jsonl(train_log_path, {
                    "event": "train_step",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "step": global_step,
                    "loss": loss_value,
                    "lr": lrs,
                    "non_padded_tokens": non_padded_tokens / total_tokens * 100
                })
                
            # 到达保存间隔时保存 last checkpoint，并在验证集上评估是否刷新 best checkpoint。
            if global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)

                val_loss = evaluate_validation_loss(accelerator, model, val_dataloader, step=global_step)
                if val_loss is not None:
                    if accelerator.is_main_process:
                        print(f"step {global_step} validation loss={val_loss:.4f}")
                        append_jsonl(train_log_path, {
                            "event": "validation",
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": val_loss
                        })
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        accelerator.wait_for_everyone()
                        save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir, filename="best.pt", val_loss=val_loss)


    # 训练结束后再同步一次，确保所有进程都完成最后一个 epoch。
    accelerator.wait_for_everyone()
    # 保存最终 checkpoint；函数内部会判断是否为主进程。
    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)
    if accelerator.is_main_process:
        append_jsonl(train_log_path, {
            "event": "train_end",
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": global_step,
            "checkpoint_dir": save_dir,
        })


if __name__ == "__main__":
    main()
