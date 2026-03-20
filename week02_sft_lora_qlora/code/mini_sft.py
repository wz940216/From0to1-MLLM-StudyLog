import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset

import logging
logger = logging.getLogger(__name__)
# ----------------------------准备模型和数据----------------------------
# 1) 加载数据文件路径（train/validation）
data_files = {
    # "train": "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_*.json"
    # "train": [
    #     "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_0.json",
    #     "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_1.json",
    #     "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_2.json",
    # ]

    "train": "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_0.json",
    "validation": "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_1.json",
}

# 2) 模型目录（本地已下载模型）
MODEL_NAME = "models/Qwen1.5-1.8B-Chat"

# 3) 读取 JSON 数据集
#    datasets.load_dataset 可以自动解析并创建 train/validation split
dataset = load_dataset("json", data_files=data_files)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# 这里为了演示，仅取前10条验证集，防止 eval 太慢
val_dataset = val_dataset.select(range(10))


# 4) 统一字段格式：instruction/input/output
#    原始 alpaca zh 数据字段是 zh_instruction/zh_input/zh_output
#    统一后方便后续 tokenizer 处理

def convert_to_zh_format(example):
    # 如果 zh_input 缺失，默认空字符串
    return {
        "instruction": example["zh_instruction"],
        "input": example.get("zh_input", ""),
        "output": example["zh_output"],
    }

zh_train_dataset = train_dataset.map(convert_to_zh_format, remove_columns=train_dataset.column_names)
zh_val_dataset = val_dataset.map(convert_to_zh_format, remove_columns=val_dataset.column_names)


# ----------------------------准备 Tokenizer、Tokenize 函数----------------------------
# 5) 初始化 tokenizer
#    trust_remote_code=True 是为了支持 Qwen 模型自定义 tokenizer class
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def make_prompt(example):
    """构建因果语言建模 prompt: 指令 +（可选输入） + 回复"""
    if example["input"] is None or example["input"] == "":
        # 无输入时只保留指令
        return f"指令: {example['instruction']}\n回复: "
    # 有输入时拼接输入
    return f"指令: {example['instruction']}\n输入: {example['input']}\n回复: "


def tokenize_fn(example):
    """把单条样本转换成模型输入ID并生成 labels"""
    prompt = make_prompt(example)
    full_text = prompt + example["output"] + tokenizer.eos_token

    # tokenizer 返回 input_ids、attention_mask 等
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        padding="max_length",
    )

    # 只计算输出部分 loss。prompt 部分不计入损失，设置为 -100。
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=1024)["input_ids"])
    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized


# 6) map 数据集进行 tokenizer 处理
#    remove_columns 删除原始列，留下 input_ids/attention_mask/labels
zh_train_dataset = zh_train_dataset.map(tokenize_fn, remove_columns=zh_train_dataset.column_names)
zh_val_dataset = zh_val_dataset.map(tokenize_fn, remove_columns=zh_val_dataset.column_names)


# 7) Data Collator: 因果语言模型不需要 MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ----------------------------评价指标----------------------------
# 8) 计算 token 准确率（忽略 label=-100）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return {"token_accuracy": 0.0}
    correct = (preds == labels) & mask
    accuracy = correct.sum() / mask.sum()
    return {"token_accuracy": float(accuracy)}


# ----------------------------模型 + lora + 训练配置----------------------------
# 9) LoRA 参数配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# 10) 加载原始因果LM模型
#    dtype=torch.bfloat16 在部分 GPU 上可提高显存效率
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)

# 关闭 cache，避免训练时梯度计算和 memory 问题
model.config.use_cache = False

# 11) 包装成 PEFT LoRA 模型
model = get_peft_model(model, peft_config)


# 12) 对齐 token id：Qwen 有些版本没有 pad_token，需要显式设置
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

if tokenizer.bos_token_id is None:
    tokenizer.bos_token = tokenizer.convert_ids_to_tokens(model.config.bos_token_id)
    tokenizer.bos_token_id = model.config.bos_token_id
if tokenizer.eos_token_id is None:
    tokenizer.eos_token = tokenizer.convert_ids_to_tokens(model.config.eos_token_id)
    tokenizer.eos_token_id = model.config.eos_token_id


# 13) 训练参数配置
training_args = TrainingArguments(
    output_dir="week2/Qwen1.5-1.8B-Chat-lora",
    gradient_checkpointing=True,
    learning_rate=1e-3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="token_accuracy",
    greater_is_better=True,

    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=50,
    eval_accumulation_steps=1,
)


# 14) Trainer 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=zh_train_dataset,
    eval_dataset=zh_val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()


# ----------------------------模型保存----------------------------
# 15) 保存 LoRA 参数和模型
model.save_pretrained("week2/Qwen1.5-1.8B-Chat-lora/model")




