# week16_multitask_tuning_summary

本周对 minillava 进行了一次完整的多任务微调实验。  

实验共分为三个阶段：  
第一阶段利用 coco caption val / VQA cal/ LLaVA-CC3M-Pretrain-595K 数据做 SFT，冻结 vision encoder 和 qwen1.5 的部分，重点训练 projector，让视觉和语言模型对齐。 

第二阶段利用多模态指令数据 LLaVA-Instruct-150K 做 SFT，开放 projector 和 qwen1.5 的部分，同时降低了 projector 的学习率，对于 llm 采用训练 LoRA 的形式，让模型更好地适应多模态指令任务。   

第三阶段利用 DPO 进行微调，提高模型在多模态指令任务上的表现。  

训练采用了四张 3090 24g显存的显卡，使用 accelerate 进行分布式训练。截止写文章为止，训练已经完成了第一阶段，用时约一天零八小时。正在进行第二阶段的训练。

为了使训练循环更完善，新增了 validation 的验证，并在验证集上保留最优模型。

```python
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


```
为了减少存储占用，在训练时，默认保存 projector 和 lora 部分的权重参数。   

```python

def should_save_param(name, param):
    """checkpoint 中保留 projector、可训练参数和 LoRA adapter 参数。"""
    name = name.lower()
    return name.startswith("projector.") or param.requires_grad or "lora_" in name or ".lora_" in name

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
```

训练 shell ：
```shell
# 阶段 1：视觉-语言对齐 / pretrain
# freeze vision encoder
# train projector
# freeze LLM
# 用 coco caption val / VQA / LLaVA-CC3M-Pretrain-595K 数据做 SFT 
# 三个epoch
accelerate launch --num_processes 4 week16_multitask_tuning_summary/code/train.py  --config week16_multitask_tuning_summary/configs/multitask_balanced_pretrain.yaml

# 阶段 2：指令微调
# freeze vision encoder
# train projector
# train LLM LoRA
# 用多模态指令数据 LLaVA-Instruct-150K
# 三个epoch
accelerate launch --num_processes 4 week16_multitask_tuning_summary/code/train.py  --config week16_multitask_tuning_summary/configs/multitask_balanced_sft.yaml

# 阶段 3：DPO
# freeze vision encoder
# freeze projector
# train LLM LoRA
accelerate launch --num_processes 4 week16_multitask_tuning_summary/code/train_dpo.py  --config week16_multitask_tuning_summary/configs/multitask_balanced_dpo.yaml
```
