# MiniLLaVA Refactored (Week09)

基于 LLaVA 设计思路的轻量多模态大模型训练与推理框架。  
项目对 [Week08](../week08_minillava_training_v1) 的代码进行了**重构**，采用 Accelerate + 模块化设计，训练流程更清晰、扩展性更好。

---

## 项目结构

```
week09_minillava_refactor/code/
├── config.yaml          # 全局配置文件（模型、数据、训练、推理参数）
├── dataset.py           # 数据集加载与 collator（LLaVA-CC3M 格式）
├── vision_encoder.py    # 视觉编码器（CLIP ViT）
├── llm_decoder.py       # 语言模型解码器 + LoRA + Projector
├── mini_llava.py        # MiniLLaVA 整体模型封装
├── train.py             # 训练脚本（Accelerate 分布式）
└── infer.py             # 推理脚本（加载 checkpoint 生成回答）
```

---

## 环境准备

```bash
# 推荐 Python 3.10+
pip install torch torchvision
pip install transformers accelerate datasets peft
pip install pillow pyyaml tqdm
```

训练多卡时额外安装 `deepspeed`（可选）并通过 `accelerate config` 配置分布式环境。

---

## 数据准备

项目使用 [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) 数据集。

```bash
# 数据集目录结构预期
dataset/LLaVA-CC3M-Pretrain-595K/
├── chat.json            # 标注文件（LLaVA 对话格式）
└── images/              # 图片目录
```

`chat.json` 中每条样本包含图片文件名与一轮 `human/gpt` 对话，脚本自动提取第一轮问答用于训练。

---

## 配置文件

[`config.yaml`](code/config.yaml) 集中管理所有参数，主要模块：

| 模块 | 关键参数 | 说明 |
|------|----------|------|
| **MINILLAVA** | 视觉编码器路径、LLM 路径、LoRA 配置、Projector 维度 | 模型结构定义 |
| **DATA** | 数据集路径、批大小、序列长度 | 数据加载 |
| **TRAINING** | 优化器类型、学习率、调度器、梯度裁剪、检查点保存 | 训练流程 |
| **INFERENCE** | 生成参数（温度、top_p/top_k、重复惩罚） | 推理生成 |
| **MISC** | 随机种子、调试模式 | 其它 |

---

## 训练

### 单卡训练

```bash
python week09_minillava_refactor/code/train.py \
    --config week09_minillava_refactor/code/config.yaml
```

### 多卡训练

```bash
accelerate launch --num_processes=4 week09_minillava_refactor/code/train.py \
    --config week09_minillava_refactor/code/config.yaml
```

### 调试模式

限制数据量快速验证训练流程是否正常：

```bash
python week09_minillava_refactor/code/train.py \
    --config week09_minillava_refactor/code/config.yaml \
    --max-samples 100
```

### 训练流程说明

1. **微调策略**：视觉编码器冻结 → Projector（随机初始化）→ LLM 加 LoRA 微调
2. **损失计算**：仅计算 **Answer 部分** 的交叉熵损失，Prompt 和图片 token 位置被 `-100` 屏蔽
3. **Accelerate 适配**：自动处理单卡/多卡/混合精度，无需手动 `.to(device)` 和 `DistributedSampler`
4. **检查点保存**：仅保存可训练参数（Projector + LoRA adapter），冻结部分从原始模型加载

#### 训练日志示例

```
epoch 1/3 step=10 loss=9.4423 lr=0.00010000
epoch 1/3 step=20 loss=8.2135 lr=0.00010000
epoch 1/3 step=500 loss=4.1256 lr=0.00009500
已保存检查点: outputs/checkpoints/step_500.pt
```

---

## 推理

使用训练好的 checkpoint 进行图文问答：

```bash
python week09_minillava_refactor/code/infer.py \
    --config week09_minillava_refactor/code/config.yaml \
    --checkpoint week08_minillava_training_v1/outputs/checkpoints/step_3000.pt \
    --image dataset/coco128/images/train2017/000000000025.jpg \
    --question "What is in the picture?"
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `week08_minillava_training_v1/code/config.yaml` | 配置文件路径 |
| `--checkpoint` | `outputs/checkpoints/step_3000.pt` | 训练好的检查点 |
| `--image` | `dataset/coco128/...` | 输入图片路径 |
| `--question` | `"What is in the picture?"` | 关于图片的问题 |

**注意**：推理脚本通过 `load_state_dict(..., strict=False)` 加载检查点，只匹配可训练参数，冻结部分使用模型初始化时的预训练权重。

---

## 模型架构

```
图片 (PIL)
  │
  ▼
┌─────────────────┐
│ CLIP Vision ViT │  ← 冻结，提取 patch 特征 (B, 196, 768)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Projector     │  ← 可训练，2 层 MLP + GELU + LayerNorm
│ (768→2048→2048) │    将视觉特征映射到 LLM embedding 空间
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Decoder    │  ← LoRA 微调（Qwen1.5-1.8B）
│ (inputs_embeds) │    图像 embedding + 文本 embedding 拼接输入
└────────┬────────┘
         │
         ▼
      回答文本
```

### 关键设计点

- **连续视觉特征**：不将图片转为离散 token，而是通过 Projector 将连续视觉特征直接映射到 LLM embedding 空间
- **序列拼接**：图片 token 放在文本 token 前面，attention mask 全 1
- **Loss 屏蔽**：CrossEntropyLoss 中图片部分和 Prompt 部分设 `-100`，只监督 Answer
- **LoRA 微调**：对大语言模型应用 LoRA（rank=8），大幅减少可训练参数量

---

## 参考

- [LLaVA: Large Language and Vision Assistant](https://arxiv.org/abs/2304.08485)
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B)
- [LLaVA-CC3M-Pretrain-595K Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate)