import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@dataclass
class SFTConfig:
    # SFT（监督微调）训练配置
    model_name_or_path: str           # 预训练模型名称或本地路径
    output_dir: str                   # 输出目录，用于保存模型与日志
    train_file: str                   # 训练数据文件（JSON）
    eval_file: Optional[str]          # 验证数据文件（JSON，可选）
    max_length: int = 1024            # 序列最大长度
    per_device_train_batch_size: int = 1  # 每个设备的训练 batch size
    per_device_eval_batch_size: int = 1   # 每个设备的验证 batch size
    num_train_epochs: int = 2             # 训练轮数
    learning_rate: float = 1e-3           # 学习率
    weight_decay: float = 0.01            # 权重衰减
    eval_steps: int = 500                 # 评估间隔（step）
    save_steps: int = 500                 # 保存间隔（step）
    logging_steps: int = 50               # 日志打印间隔（step）
    gradient_checkpointing: bool = True   # 是否开启梯度检查点（省显存）
    warmup_steps: int = 0                 # 预热步数


def setup_logging() -> None:
    # 设置日志格式和级别
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # 降低第三方库日志等级，避免过多输出
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)


def make_prompt(instruction: str, input_text: str = "") -> str:
    """构造简单的中文指令-回复形式的 prompt。"""
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        # 包含输入时的模板
        return f"指令: {instruction}\n输入: {input_text}\n回复: "
    # 不包含输入时的模板
    return f"指令: {instruction}\n回复: "


def build_tokenizer_and_model(model_name: str):
    """加载分词器和基础 CausalLM 模型。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # 保证 tokenizer 一定有 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            # 若有 eos_token，则复用为 pad_token
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # 否则直接抛错
            raise ValueError("Tokenizer has no eos_token and no pad_token, cannot set pad token.")

    # 若 tokenizer 没有 bos_token，但模型配置里有，则同步设置
    if tokenizer.bos_token_id is None and model.config.bos_token_id is not None:
        tokenizer.bos_token_id = model.config.bos_token_id
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens(model.config.bos_token_id)

    # 若 tokenizer 没有 eos_token，但模型配置里有，则同步设置
    if tokenizer.eos_token_id is None and model.config.eos_token_id is not None:
        tokenizer.eos_token_id = model.config.eos_token_id
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(model.config.eos_token_id)

    return tokenizer, model


def load_json_datasets(train_file: str, eval_file: Optional[str] = None) -> DatasetDict:
    """从 JSON 文件中加载训练与可选的验证数据集，返回 DatasetDict。"""
    data_files = {"train": train_file}
    if eval_file is not None:
        data_files["validation"] = eval_file
    dataset = load_dataset("json", data_files=data_files)
    if eval_file is None:
        logger.warning("No eval file provided. Training set only.")
    return dataset


def normalize_example(example: Dict[str, Union[str, None]]) -> Dict[str, str]:
    """将数据样本字段统一归一到 {instruction, input, output}。"""
    # 若已有标准字段 instruction/output，则直接使用
    if "instruction" in example and "output" in example:
        return {
            "instruction": example["instruction"] or "",
            "input": example.get("input", "") or "",
            "output": example["output"] or "",
        }
    # 否则认为是中文 Alpaca 格式，使用 zh_* 字段
    return {
        "instruction": example.get("zh_instruction", "") or "",
        "input": example.get("zh_input", "") or "",
        "output": example.get("zh_output", "") or "",
    }


def tokenize_function(
    example: Dict[str, Union[str, None]],
    tokenizer,
    max_length: int,
) -> Dict[str, np.ndarray]:
    """对单条样本进行分词，并将 prompt 部分的标签 mask 为 -100。"""
    # 先归一化字段
    example = normalize_example(example)
    # 构造指令类 prompt
    prompt = make_prompt(example["instruction"], example["input"])
    # 完整输入 = prompt + 输出 + 结尾 token
    full_text = prompt + example["output"] + tokenizer.eos_token

    # 对完整文本进行分词、截断和 padding
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    # 对 prompt 单独分词以得到 prompt 的 token 数
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
    # labels 初始化为 input_ids 的拷贝
    labels = tokenized["input_ids"].copy()
    # 将 prompt 对应位置的 label 置为 -100，避免在损失中计算
    for idx in range(min(len(prompt_ids), len(labels))):
        labels[idx] = -100
    tokenized["labels"] = labels
    return tokenized


def prepare_datasets(dataset: DatasetDict, tokenizer, max_length: int = 1024) -> DatasetDict:
    """对 train / validation 切分进行分词映射，返回新的 DatasetDict。"""
    result = {}
    for split in dataset:
        # 需要移除原始列，只保留 tokenized 字段
        remove_columns = dataset[split].column_names
        result[split] = dataset[split].map(
            lambda ex: tokenize_function(ex, tokenizer, max_length),
            remove_columns=remove_columns,
            batched=False,   # 逐样本处理，便于使用 normalize_example
            num_proc=1,      # 单进程，以避免多进程问题
        )
    return DatasetDict(result)


def compute_metrics(eval_pred):
    """计算 token 级别准确率，忽略 label 为 -100 的位置。"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)   # 取每个位置的最大概率 token 作为预测
    mask = labels != -100                # 只统计未 mask 的位置
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}
    correct = (preds == labels) & mask
    return {"token_accuracy": float(correct.sum() / mask.sum())}


def train_sft(config: SFTConfig):
    """使用 Trainer 进行 LoRA SFT 训练。"""
    logger.info("Loading tokenizer and model from %s", config.model_name_or_path)
    tokenizer, model = build_tokenizer_and_model(config.model_name_or_path)

    logger.info("Loading datasets")
    dataset = load_json_datasets(config.train_file, config.eval_file)

    if "validation" not in dataset:
        logger.warning("No validation split found, skipping eval dataset.")

    # 先统一字段格式，避免后面处理不一致
    dataset = dataset.map(lambda ex: normalize_example(ex), remove_columns=dataset["train"].column_names)

    # 配置 LoRA 参数
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 任务类型为自回归语言模型
        inference_mode=False,          # 训练模式
        r=8,                           # LoRA rank
        lora_alpha=32,                 # LoRA scaling 系数
        lora_dropout=0.1,              # LoRA dropout
    )
    # 将模型转为 PEFT LoRA 模型
    model = get_peft_model(model, lora_config)
    # 关闭缓存以适配梯度检查点等训练设置
    model.config.use_cache = False

    # 对数据集进行分词与 labels 构造
    tokenized_dataset = prepare_datasets(dataset, tokenizer, max_length=config.max_length)

    # 构造数据整理器（LM 任务，不使用 MLM）
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 配置 Trainer 的训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_checkpointing=config.gradient_checkpointing,
        # 若存在 validation 切分，则按 step 评估；否则不评估
        eval_strategy="steps" if "validation" in tokenized_dataset else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",          # 按 step 保存模型
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        # 若有验证集，则在训练结束加载最佳模型
        load_best_model_at_end=True if "validation" in tokenized_dataset else False,
        metric_for_best_model="token_accuracy",  # 用 token 准确率作为最优指标
    )

    # 构造 Trainer 对象
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", None),
        processing_class=tokenizer,  # 传入 tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics if "validation" in tokenized_dataset else None,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving LoRA checkpoint and tokenizer")
    # 保存 LoRA 适配模块权重
    model.save_pretrained(os.path.join(config.output_dir, "lora_model"))
    # 保存 tokenizer
    tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))
    logger.info("Training finished. Models saved to %s", config.output_dir)


def infer(
    base_model_name: str,
    lora_weights: str,
    prompt: str,
    max_new_tokens: int = 250,
    device: str = "cuda",
):
    """使用基础模型 + LoRA 权重进行推理。"""
    # 加载基础模型与 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
    # 将 LoRA 权重加载到基础模型上
    model = PeftModel.from_pretrained(base_model, lora_weights)
    model.eval().to(device)  # 设置为 eval 模式并移动到指定设备

    # 将 prompt 编码为模型输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 生成文本
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # 丢弃 prompt 部分，只取模型新生成的 token
    gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:].detach().cpu().numpy()
    # 将 token id 解码为文本（跳过特殊 token）
    generated = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return generated.strip()  # 去除首尾空白字符后返回


def main():
    # 初始化日志
    setup_logging()
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description="Reusable SFT + LoRA training and inference script.")
    # 创建子命令解析器：train / infer
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 训练子命令
    train_parser = subparsers.add_parser("train", help="Train LoRA SFT.")
    train_parser.add_argument("--model_name_or_path", required=True)            # 模型名称或路径
    train_parser.add_argument("--train_file", required=True)                    # 训练数据文件
    train_parser.add_argument("--eval_file", required=False)                    # 验证数据文件
    train_parser.add_argument("--output_dir", required=True)                    # 输出目录
    train_parser.add_argument("--num_train_epochs", type=int, default=2)        # 训练轮数
    train_parser.add_argument("--learning_rate", type=float, default=1e-3)      # 学习率
    train_parser.add_argument("--per_device_train_batch_size", type=int, default=1)  # 训练 batch size
    train_parser.add_argument("--per_device_eval_batch_size", type=int, default=1)   # 验证 batch size
    train_parser.add_argument("--max_length", type=int, default=1024)           # 最大序列长度
    train_parser.add_argument("--eval_steps", type=int, default=500)            # 评估间隔
    train_parser.add_argument("--save_steps", type=int, default=500)            # 保存间隔
    train_parser.add_argument("--logging_steps", type=int, default=50)          # 日志间隔

    # 推理子命令
    infer_parser = subparsers.add_parser("infer", help="Run inference with LoRA.")
    infer_parser.add_argument("--base_model_name", required=True)               # 基础模型名称或路径
    infer_parser.add_argument("--lora_weights", required=True)                  # LoRA 权重目录
    infer_parser.add_argument("--instruction", required=True)                   # 指令内容
    infer_parser.add_argument("--input_text", default="")                       # 可选输入内容
    infer_parser.add_argument("--max_new_tokens", type=int, default=250)        # 生成最大新 token 数

    # 解析命令行参数
    args = parser.parse_args()

    if args.command == "train":
        # 将命令行参数组装成 SFTConfig 配置
        config = SFTConfig(
            model_name_or_path=args.model_name_or_path,
            output_dir=args.output_dir,
            train_file=args.train_file,
            eval_file=args.eval_file,
            max_length=args.max_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            gradient_checkpointing=True,
        )
        # 启动训练
        train_sft(config)
    else:
        # 构造推理用 prompt
        prompt = make_prompt(args.instruction, args.input_text)
        print("Prompt:", prompt)
        # 运行推理
        response = infer(
            base_model_name=args.base_model_name,
            lora_weights=args.lora_weights,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n--- RESPONSE ---\n")
        print(response)


if __name__ == "__main__":
    # 当脚本作为主程序运行时，执行 main()
    main()
