# 第 12 周：整理 MiniLLaVA 工程与文档

本周重点不是继续堆模型能力，而是把前面几周的实验代码整理成更接近开源工程的形态：目录清楚、入口明确、配置可复用、日志可追踪、文档能让别人快速跑起来。

当前第 12 周目录保留四类内容：

- `code/`：核心源码，包括模型、数据集、训练和推理入口。
- `configs/`：不同任务组合和运行环境的配置样例。
- `utils/`：数据格式转换工具。
- `outputs/`：checkpoint、上下文和日志等运行产物。


- 工程目录结构。
- MiniLLaVA 架构：CLIP vision encoder、projector、LLM decoder。
- LLaVA conversations 与 OpenAI chat messages 两种数据格式。
- 训练命令、推理命令、多轮上下文保存方式。
- checkpoint、训练日志、推理日志的默认输出位置。
- 端到端 smoke test 建议。


训练脚本新增 JSONL 日志：

- `train_start`：记录配置路径、epoch 数、总步数、调试样本数。
- `train_step`：按 `LOG_STEPS` 记录 loss 和 lr。
- `train_end`：记录最终 step 和 checkpoint 目录。

推理脚本新增 JSONL 日志：

- 记录图片路径、问题、回答、上下文轮数、是否交互模式。
- 默认读取配置里的 `INFERENCE.LOG_FILE`。
- 命令行可用 `--log-file` 覆盖。


新增三个配置样例：

- `configs/projector_debug.yaml`：小样本调试配置，适合先跑通训练循环。
- `configs/multitask_balanced.yaml`：COCO caption 与 VQA 均衡多任务训练。
- `configs/caption_only_cpu.yaml`：CPU smoke test 配置，只建议配合 `--max-samples` 使用。

这些配置把模型大小、任务组合、日志目录、checkpoint 目录拆开，避免每次实验都直接改默认配置。


前三个月的主要收获：

- 从单轮 VQA 逐步扩展到多轮图文对话，理解了数据格式对模型训练接口的影响。
- 掌握了 MiniLLaVA 的核心拼接方式：用 `<image>` token 作为视觉 embedding 插入位置。
- 熟悉了冻结底座、训练 projector、加入 LoRA adapter 的轻量微调路径。
- 开始把多任务数据混合、采样比例、collator label mask 等工程细节纳入训练流程。

当前不足：

- 评估体系还偏弱，目前主要依赖人工推理观察，缺少固定验证集指标。
- 日志还只是基础 JSONL，后续可以接入 TensorBoard、WandB 或简单可视化脚本。
- checkpoint 管理还可以继续完善，例如自动清理旧 checkpoint、记录 git commit、记录数据版本。
- 推理输出格式虽然加入 JSON 约束，但模型未必稳定遵守，需要更多格式校验和失败重试策略。


