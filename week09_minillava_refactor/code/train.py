import argparse
import os
import random

import torch
import yaml
# Accelerator 负责把普通 PyTorch 训练脚本扩展到单卡、多卡、混合精度等场景。
from accelerate import Accelerator
# accelerate_set_seed 会额外处理分布式训练中的随机种子同步。
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
# 复用 transformers 提供的常见学习率调度器，避免手写 warmup/cosine 逻辑。
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

# 数据集负责读取图片和问答文本，Collator 负责在 batch 阶段 tokenize 并构造 labels。
from dataset import LlavaCollator, LlavaPretrainDataset
# MiniLlavaModel 封装视觉编码器、投影层和语言模型解码器。
from mini_llava import MiniLlavaModel


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
    # 训练配置中 OPTIMIZER 字段决定优化器类型、学习率和权重衰减等参数。
    optim_config = config["TRAINING"]["OPTIMIZER"]
    # 过滤掉被冻结的视觉编码器或语言模型参数，只训练当前允许更新的部分。
    params = [p for p in model.parameters() if p.requires_grad]
    optim_type = optim_config["TYPE"].lower()
    # AdamW 是大语言模型微调中最常用的优化器，带 decoupled weight decay。
    if optim_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(optim_config["LR"]),
            weight_decay=float(optim_config["WEIGHT_DECAY"]),
            betas=tuple(optim_config["BETAS"])
        )
    # Adam 不使用 AdamW 的解耦权重衰减，适合简单调试或对比实验。
    if optim_type == "adam":
        return torch.optim.Adam(params, lr=float(optim_config["LR"]))
    # SGD 一般不用于 LLM 微调，但保留入口便于实验。
    if optim_type == "sgd":
        return torch.optim.SGD(params, lr=float(optim_config["LR"]), momentum=0.9)
    raise ValueError(f"不支持的优化器类型: {optim_type}")


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
    """checkpoint 中保留可训练参数和 LoRA adapter 参数。"""
    name = name.lower()
    return param.requires_grad or "lora_" in name or ".lora_" in name


def save_checkpoint(accelerator, model, optimizer, scheduler, step, save_dir):
    """保存训练检查点，包含 projector、可训练参数和优化器状态。"""
    # 多卡训练时每个进程都会执行代码；只让主进程写文件，避免多个进程同时覆盖同一路径。
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
    # prepare 后的 model 可能被 DDP/FSDP 等包装；保存前要取回原始模型对象。
    unwrapped_model = accelerator.unwrap_model(model)
    # 只保存开启梯度的参数和 LoRA adapter 参数，冻结的基础模型权重由初始化模型提供。
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


def main():
    # 命令行参数只保留配置路径和调试样本数，其他训练参数统一放在 YAML 中管理。
    parser = argparse.ArgumentParser(description="MiniLLaVA 微调脚本")
    parser.add_argument("--config", default="week08_minillava_training_v1/code/config.yaml", help="训练配置文件路径")
    parser.add_argument("--max-samples", type=int, default=1000, help="调试时只取前 N 条数据")
    args = parser.parse_args()

    # Accelerator 会根据 accelerate launch 的启动方式自动识别进程数、设备和分布式后端。
    accelerator = Accelerator()
    config = load_config(args.config)
    # 保留原本的 PyTorch 随机种子设置。
    set_seed(int(config["MISC"]["SEED"]))
    # 再使用 Accelerate 的种子工具，保证多进程场景下每个进程的随机状态可控。
    accelerate_set_seed(int(config["MISC"]["SEED"]))

    # 初始化 MiniLLaVA，并切换到训练模式，启用 dropout 等训练期行为。
    model = MiniLlavaModel(args.config)
    model.train()

    # 从配置中读取训练数据路径、图片目录、标注文件、batch size 等数据相关参数。
    train_config = config["DATA"]["TRAIN_DATASET"]
    dataset = LlavaPretrainDataset(
        dataset_path=train_config["PATH"],
        image_dir=train_config["IMAGE_DIR"],
        annotation_file=train_config["ANNOTATION_FILE"],
        max_samples=args.max_samples
    )
    # Collator 在 DataLoader 拼 batch 时进行 tokenizer、padding，并构造只监督 answer 的 labels。
    collator = LlavaCollator(
        tokenizer=model.language_decoder.tokenizer,
        max_length=int(train_config["MAX_LENGTH"])
    )
    # DataLoader 仍按普通 PyTorch 写法创建；后面 accelerator.prepare 会自动处理多卡分片。
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(train_config["NUM_WORKERS"]),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )

    # 先在原始 model 上创建优化器，这样 optimizer 能拿到正确的可训练参数列表。
    optimizer = build_optimizer(model, config)
    num_epochs = int(config["TRAINING"]["SCHEDULER"]["NUM_EPOCHS"])
    # prepare 会把 model 放到正确设备，并在多卡时包装为分布式模型；
    # dataloader 也会被切成每个进程各自负责的一份数据。
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # prepare 之后的 dataloader 长度是当前进程实际迭代步数，用它计算 scheduler 总步数更适合多卡。
    total_steps = len(dataloader) * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)
    # scheduler 依赖已经 prepare 过的 optimizer，因此在 optimizer prepare 之后创建并交给 accelerator。
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # 日志、保存、梯度裁剪等训练控制参数。
    log_steps = int(config["TRAINING"]["LOGGING"]["LOG_STEPS"])
    save_steps = int(config["TRAINING"]["CHECKPOINT"]["SAVE_STEPS"])
    save_dir = config["TRAINING"]["CHECKPOINT"]["SAVE_DIR"]
    max_norm = float(config["TRAINING"]["GRAD_CLIP"]["MAX_NORM"])

    # global_step 记录当前进程执行的优化步数；多卡下各进程同步前进。
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
            
            # 只让全局主进程打印日志，避免多进程重复输出相同 step。
            if global_step % log_steps == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch + 1}/{num_epochs} step={global_step} loss={loss_value:.4f} lr={lr:.8f}")
                
            # 到达保存间隔时，先等待所有进程到同一步，再由主进程写 checkpoint。
            if global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)

    # 训练结束后再同步一次，确保所有进程都完成最后一个 epoch。
    accelerator.wait_for_everyone()
    # 保存最终 checkpoint；函数内部会判断是否为主进程。
    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)


if __name__ == "__main__":
    main()
