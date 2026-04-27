import argparse
import os
import random

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from dataset import LlavaCollator, LlavaPretrainDataset
from mini_llava import MiniLlavaModel


def set_seed(seed):
    """固定随机种子，方便复现实验结果。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    """读取训练配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_optimizer(model, config):
    """根据配置创建优化器，只更新 requires_grad=True 的参数。"""
    optim_config = config["TRAINING"]["OPTIMIZER"]
    params = [p for p in model.parameters() if p.requires_grad]
    optim_type = optim_config["TYPE"].lower()
    if optim_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(optim_config["LR"]),
            weight_decay=float(optim_config["WEIGHT_DECAY"]),
            betas=tuple(optim_config["BETAS"])
        )
    if optim_type == "adam":
        return torch.optim.Adam(params, lr=float(optim_config["LR"]))
    if optim_type == "sgd":
        return torch.optim.SGD(params, lr=float(optim_config["LR"]), momentum=0.9)
    raise ValueError(f"不支持的优化器类型: {optim_type}")


def build_scheduler(optimizer, config, total_steps):
    """创建学习率调度器。"""
    sched_config = config["TRAINING"]["SCHEDULER"]
    warmup_steps = int(sched_config["WARMUP_STEPS"])
    sched_type = sched_config["TYPE"].lower()
    if sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_type == "constant":
        return None
    raise ValueError(f"不支持的调度器类型: {sched_type}")


def save_checkpoint(model, optimizer, scheduler, step, save_dir):
    """保存训练检查点，包含 projector、可训练参数和优化器状态。"""
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None
        },
        ckpt_path
    )
    print(f"已保存检查点: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="MiniLLaVA 微调脚本")
    parser.add_argument("--config", default="week07_minillava_design_data/code/config.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="调试时只取前 N 条数据")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["MISC"]["SEED"]))

    model = MiniLlavaModel(args.config)
    model.train()

    train_config = config["DATA"]["TRAIN_DATASET"]
    dataset = LlavaPretrainDataset(
        dataset_path=train_config["PATH"],
        image_dir=train_config["IMAGE_DIR"],
        annotation_file=train_config["ANNOTATION_FILE"],
        max_samples=args.max_samples
    )
    collator = LlavaCollator(
        tokenizer=model.language_decoder.tokenizer,
        max_length=int(train_config["MAX_LENGTH"])
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(train_config["NUM_WORKERS"]),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )

    optimizer = build_optimizer(model, config)
    num_epochs = int(config["TRAINING"]["SCHEDULER"]["NUM_EPOCHS"])
    total_steps = len(dataloader) * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)

    log_steps = int(config["TRAINING"]["LOGGING"]["LOG_STEPS"])
    save_steps = int(config["TRAINING"]["CHECKPOINT"]["SAVE_STEPS"])
    save_dir = config["TRAINING"]["CHECKPOINT"]["SAVE_DIR"]
    max_norm = float(config["TRAINING"]["GRAD_CLIP"]["MAX_NORM"])

    global_step = 0
    for epoch in range(num_epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{num_epochs}")
        for batch in progress:
            outputs = model(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % log_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"step={global_step} loss={loss.item():.4f} lr={lr:.8f}")

            if global_step % save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, save_dir)

    save_checkpoint(model, optimizer, scheduler, global_step, save_dir)


if __name__ == "__main__":
    main()
