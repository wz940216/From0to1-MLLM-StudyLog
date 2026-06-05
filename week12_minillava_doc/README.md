# Week 12: MiniLLaVA 工程整理与文档

本目录是第 12 周的 MiniLLaVA 工程化版本，目标是把前几周逐步实现的视觉编码器、语言解码器、多轮数据格式、训练脚本和推理脚本整理成一个更容易复用的小型开源工程。

## 目录结构

```text
week12_minillava_doc/
├── README.md                  # 对外说明文档
├── note.md                    # 第 12 周复盘记录
├── code/                      # 核心训练和推理代码
│   ├── config.yaml            # 默认多任务配置
│   ├── dataset.py             # LLaVA/OpenAI chat 数据读取与 collator
│   ├── infer.py               # 单轮/多轮推理脚本
│   ├── llm_decoder.py         # LLM、LoRA、projector 相关模块
│   ├── mini_llava.py          # MiniLLaVA 总模型
│   ├── train.py               # 训练入口
│   └── vision_encoder.py      # CLIP 视觉编码器
├── configs/                   # 可复用配置样例
│   ├── caption_only_cpu.yaml
│   ├── multitask_balanced.yaml
│   └── projector_debug.yaml
├── outputs/                   # checkpoint、上下文和日志输出
└── utils/                     # 数据格式转换工具
```

## 模型架构

MiniLLaVA 由三部分组成：

1. `VisionEncoder`：加载 CLIP ViT，把输入图片编码成 patch-level 视觉特征。
2. `Projector`：把 CLIP 输出维度映射到语言模型 hidden size，使视觉 token 可以接入 LLM。
3. `LLMDecoder`：加载 Qwen1.5 等语言模型，可冻结底座并通过 LoRA 训练少量 adapter 参数。

训练时，文本 prompt 中的 `<image>` token 会被替换成整段视觉 patch embedding。`dataset.py` 会把 user 部分、padding 部分和图像 patch 的 label 置为 `-100`，只监督 assistant 的回答。

## 数据格式

当前数据集同时兼容两类标注：

- LLaVA `conversations` 格式：`human/gpt` 轮次。
- OpenAI chat `messages` 格式：第一轮 user 可包含 `image` block 和 `text` block，后续轮次可以是纯文本。

推荐新数据使用 OpenAI chat 格式，便于后续扩展多图、视频、tool call 或 grounding 数据。

## 训练

默认配置位于 `code/config.yaml`，路径默认指向 `week12_minillava_doc`。

```bash
python week12_minillava_doc/code/train.py   --config week12_minillava_doc/code/config.yaml
```

调试时可以先限制样本数：

```bash
python week12_minillava_doc/code/train.py   --config week12_minillava_doc/configs/projector_debug.yaml   --max-samples 16
```

训练 checkpoint 默认写入：

```text
week12_minillava_doc/outputs/checkpoints/
```

训练日志默认写入 JSONL：

```text
week12_minillava_doc/outputs/logs/train.jsonl
```

每条日志包含 `event`、`time`、`step`、`loss`、`lr` 等字段，后续可以直接用 pandas 或 jq 分析。

## 推理

单轮推理：

```bash
python week12_minillava_doc/code/infer.py   --config week12_minillava_doc/code/config.yaml   --checkpoint week12_minillava_doc/outputs/checkpoints/step_2109.pt   --image dataset/coco128/images/train2017/000000000025.jpg   --question "Please describe this image."
```

多轮推理并保存上下文：

```bash
python week12_minillava_doc/code/infer.py   --config week12_minillava_doc/code/config.yaml   --checkpoint week12_minillava_doc/outputs/checkpoints/step_2109.pt   --image dataset/coco128/images/train2017/000000000025.jpg   --context-file week12_minillava_doc/outputs/context/demo_chat.json   --interactive
```

推理日志默认写入：

```text
week12_minillava_doc/outputs/logs/infer.jsonl
```

也可以通过 `--log-file` 覆盖日志路径。

## 配置样例

`configs/projector_debug.yaml`：小样本快速调试，冻结视觉编码器和 LLM，适合先检查数据和训练循环。

`configs/multitask_balanced.yaml`：COCO caption 与 VQA 按 0.5/0.5 混合采样，适合验证多任务训练。

`configs/caption_only_cpu.yaml`：CPU smoke test 配置，只建议配合 `--max-samples` 跑通流程。

## demo

```bash
python week12_minillava_doc/code/train.py   --config week12_minillava_doc/configs/projector_debug.yaml   --max-samples 8

python week12_minillava_doc/code/infer.py   --config week12_minillava_doc/configs/projector_debug.yaml   --checkpoint week12_minillava_doc/outputs/checkpoints/projector_debug/step_8.pt   --image dataset/coco128/images/train2017/000000000025.jpg   --question "What is in the image?"
```

