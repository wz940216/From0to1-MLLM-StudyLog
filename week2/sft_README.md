# week2/sft.py 使用指南

本文件给出一键复现训练、日志说明、checkpoint结构以及推理示例。

## 1. 目录说明

- `week2/sft.py`：可复用 SFT + LoRA 脚本（train / infer）
- `week2/sft_config.json`：示例配置
- `week2/Qwen1.5-1.8B-Chat-lora/`：训练输出目录（checkpoint、模型、tokenizer）

## 2. 环境安装

推荐使用 conda 环境：

```bash
conda create -n mllm python=3.11 -y
conda activate mllm
pip install -U pip
pip install transformers datasets accelerate peft safetensors
```

> 如果你使用 `Qwen` 的 `trust_remote_code=True` 模型，确保 transformers 版本 >= 4.34，PEFT 版本 >= 0.6。

## 3. 配置文件（sft_config.json）

下面是最简配置（已在 `week2/sft_config.json`）：

```json
{
  "model_name_or_path": "models/Qwen1.5-1.8B-Chat",
  "train_file": "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_0.json",
  "eval_file": "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_1.json",
  "output_dir": "week2/Qwen1.5-1.8B-Chat-lora",
  "max_length": 1024,
  "per_device_train_batch_size": 1,
  "per_device_eval_batch_size": 1,
  "num_train_epochs": 2,
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "eval_steps": 500,
  "save_steps": 500,
  "logging_steps": 50,
  "gradient_checkpointing": true
}
```

## 4. 一键训练

### 4.1 直接命令参数方式

```bash
python week2/sft.py train \
  --model_name_or_path models/Qwen1.5-1.8B-Chat \
  --train_file dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_0.json \
  --eval_file dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_1.json \
  --output_dir week2/Qwen1.5-1.8B-Chat-lora \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --max_length 1024 \
  --learning_rate 0.001 \
  --eval_steps 500 \
  --save_steps 500 \
  --logging_steps 50
```

### 4.2 读取 JSON 配置快速复现

```bash
python week2/sft.py train \
  --model_name_or_path $(jq -r '.model_name_or_path' week2/sft_config.json) \
  --train_file $(jq -r '.train_file' week2/sft_config.json) \
  --eval_file $(jq -r '.eval_file' week2/sft_config.json) \
  --output_dir $(jq -r '.output_dir' week2/sft_config.json) \
  --num_train_epochs $(jq -r '.num_train_epochs' week2/sft_config.json) \
  --per_device_train_batch_size $(jq -r '.per_device_train_batch_size' week2/sft_config.json) \
  --per_device_eval_batch_size $(jq -r '.per_device_eval_batch_size' week2/sft_config.json) \
  --max_length $(jq -r '.max_length' week2/sft_config.json) \
  --learning_rate $(jq -r '.learning_rate' week2/sft_config.json) \
  --eval_steps $(jq -r '.eval_steps' week2/sft_config.json) \
  --save_steps $(jq -r '.save_steps' week2/sft_config.json) \
  --logging_steps $(jq -r '.logging_steps' week2/sft_config.json)
```

> 如果环境没有 `jq`，直接用参数方式更稳定。

## 5. 日志与 checkpoint 结构（Trainer 输出）

训练输出目录 `week2/Qwen1.5-1.8B-Chat-lora/` 典型内容：

- `checkpoint-500/`, `checkpoint-1000/` 等：中间检查点
- `trainer_state.json`、`trainer_config.json`、`runs/`：训练状态与日志
- `pytorch_model.bin`（如果保存全模型）
- `lora_model/`：LoRA 训练结果（`adapter_model.bin`/`pytorch_model.bin`）
- `tokenizer/`：tokenizer 配置文件

### 5.1 一般目录结构

```
week2/Qwen1.5-1.8B-Chat-lora/
├── checkpoint-500/
├── checkpoint-1000/
├── lora_model/
│   ├── adapter_model.bin
│   └── ...
├── tokenizer/
│   ├── tokenizer.json
│   └── ...
├── trainer_state.json
├── trainer_config.json
└── runs/
    └── ...
```

## 6. 推理示例

### 6.1 直接调用 `infer` 子命令

```bash
python week2/sft.py infer \
  --base_model_name models/Qwen1.5-1.8B-Chat \
  --lora_weights week2/Qwen1.5-1.8B-Chat-lora/lora_model \
  --instruction "给出三个保持健康的建议。" \
  --input_text ""
```

### 6.2 在脚本里调用 `infer()`

```python
from week2.sft import infer
prompt = "指令: 请简述量子纠缠的核心概念。\n回复: "
print(infer("models/Qwen1.5-1.8B-Chat", "week2/Qwen1.5-1.8B-Chat-lora/lora_model", prompt))
```

## 7. 常见问题

- `ModuleNotFoundError: No module named 'datasets'`：请安装 `pip install datasets`
- `CUDA out of memory`：减小 `per_device_train_batch_size` 或 `max_length`，开启 `gradient_checkpointing`
- `tokenizer.pad_token_id is None`：脚本已自动补齐 eos/pad，如果仍异常请手动设置 `tokenizer.pad_token = tokenizer.eos_token`。
