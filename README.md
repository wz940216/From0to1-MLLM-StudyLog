
# From0to1-MLLM-StudyLog

> 从 0 到 1 自学多模态大模型（MLLM）的 24 周实践与工程记录

本仓库是作者从零系统学习大语言模型（LLM）与多模态大模型（MLLM）的完整过程记录，周期约 24 周(持续更新中)，包含：

- 环境搭建、LLM 基础、SFT/LoRA/QLoRA 实战
- CLIP / BLIP / BLIP-2 / LLaVA 等代表性多模态架构理解与复现
- 自建 mini-LLaVA（mini 多模态模型）的训练、重构与多任务扩展
- 指令对齐、安全策略与多轮对话等工程问题
- 推理加速（vLLM / TensorRT / OpenVINO）与服务部署
- 最终落地一个代表性多模态应用（如文档多模态问答系统）

仓库定位为「个人自学 + 实战工程笔记」，不是正式课程，但力求做到：

- **路径完整**：从“能跑一个 LLM demo”到“能从 0 实现一个 mini 多模态系统再部署”  
- **工程可复现**：代码、配置和实验过程尽量可复用  
- **真实透明**：包含踩坑、失败实验和阶段性复盘

---

## 仓库结构

```bash
.
├── README.md
├── dataset               # 数据集本地目录
├── docs/
│   ├── roadmap.md        # 本 README 中的 24 周计划
│   ├── refs.md           # 论文 / 教程 / 博客等资料索引
│   └── notes/            # 跨周的总结型笔记（对齐、安全、部署等专题）
├── models                # 模型本地目录
├── week01_env_llm_transformers/
├── week02_sft_lora_qlora/
├── week03_mllm_overview_llava_demo/
├── week04_clip_retrieval/
├── week05_blip_blip2_caption/
├── week06_clip_blip_web_demo/
├── week07_minillava_design_data/
├── week08_minillava_training_v1/
├── week09_minillava_refactor/
├── week10_multitask_data/
├── week11_multiturn_dialogue/
├── week12_minillava_doc/
├── week13_chat_format_alignment/
├── week14_dialogue_stability_output_control/
├── week15_safety_alignment/
├── week16_multitask_tuning_summary/
├── week17_llm_inference_vllm/
├── week18_vision_tensorrt_openvino/
├── week19_end2end_service/
├── week20_benchmark_optimization/
├── week21_project_design/
├── week22_project_core_impl/
├── week23_project_optimize/
└── week24_project_release_summary/
```
---

## 学习与实践路线概览（24 周）

下面是本仓库对应的整体路线，只是目标与内容的大纲；  
每一周的具体笔记、代码和脚本会放在对应的 `weekXX_*/` 目录中。

---

### 第 1–4 周：基础环境 + LLM & 多模态概览 + CLIP

#### 第 1 周：环境搭建 & Transformers 入门

- 搭建 WSL2（Ubuntu）+ Conda + CUDA + PyTorch 环境
- 安装 `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes` 等基础依赖
- 用 Qwen / Llama 等模型写简单推理脚本和 CLI 对话脚本
- 输出：环境配置文档 + 第一个 LLM CLI demo

#### 第 2 周：SFT + LoRA / QLoRA 入门

- 理解 LoRA / QLoRA 基本原理与使用场景
- 搭建基于 PEFT 的微调 Pipeline
- 准备一份指令数据（如 alpaca 格式），在小模型上进行一次 SFT
- 对比微调前后模型行为，沉淀可复用的 SFT 脚本和日志

#### 第 3 周：多模态概览 + 跑通一个 LLaVA Demo

- 阅读多模态 LLM 综述，建立整个 MLLM 的全景认知
- 粗读 CLIP、BLIP-2、LLaVA、MiniGPT-4 等方法的结构差异
- 在本地以 4bit/8bit 形式跑通 `llava-hf/llava-1.5-7b-hf` 推理
- 写一个简化的图文问答脚本，并用笔记/流程图梳理 Vision Encoder / Projector / LLM 的数据流

#### 第 4 周：CLIP 入门 + 图文检索

- 深入阅读 CLIP 方法部分（对比学习、损失函数）
- 使用开源 CLIP 模型做图像 + 文本相似度计算
- 实现一个最小可用的图文检索 demo + 简单零样本分类
- 输出：CLIP 检索 & 零样本分类小工具 + 性能记录

---

### 第 5–8 周：BLIP/BLIP-2 + 图像检索 Web + mini 多模态训练

#### 第 5 周：BLIP / BLIP-2 Caption

- 理解 BLIP / BLIP-2 的整体架构与 Caption 思路
- 跑通 `Salesforce/blip-image-captioning-base` 和 `Salesforce/blip2-flan-t5-xl`
- 在一小部分 COCO Caption 上 finetune BLIP，体验中文/多语言 Caption
- 输出：Caption 推理 & 简单微调脚本 + 实验记录

#### 第 6 周：组合 CLIP + BLIP 做 Web Demo

- 使用 Gradio / FastAPI 做一个小 Web 应用：
  - 上传图片 -> BLIP Caption -> 显示结果
  - 文本 Query -> CLIP 检索图库图片 -> 显示 Top-K
- 优化性能（embedding 预计算、batch 推理），并整理 demo 的 README

#### 第 7 周：mini 多模态训练准备（数据 & 架构）

- 阅读 LLaVA 训练文档，理解视觉指令数据格式和 loss 设计
- 选取一小部分 LLaVA 数据作为训练集
- 设计自己的 mini-LLaVA：如 CLIP-ViT-B/16 + Qwen1.5-1.8B + MLP Projector
- 实现 `MiniLlavaModel` 的骨架和配置文件（JSON/YAML）

#### 第 8 周：实现 mini-LLaVA 训练脚本 & 小规模训练

- 实现多模态 DataLoader（image + conversation -> token 序列）
- 写训练 loop，接入 LoRA/QLoRA，只训练 projector + LLM LoRA
- 跑一次短训练，观察 loss 变化，并实现推理脚本
- 输出：第一个可跑通的 mini-LLaVA v1 + 训练 & 推理脚本

---

### 第 9–12 周：完善 mini-LLaVA & 多任务训练

#### 第 9 周：深入理解 LLaVA & 模型重构

- 深入对比 mini 实现与 LLaVA 官方实现（VisionTower / Projector 等）
- 对 mini-LLaVA 工程做模块化重构，加入配置管理
- 编写简单单元测试（shape 校验、前向检查），并再次在小数据集上验证

#### 第 10 周：多任务数据（Caption + VQA + QA）

- 下载 VQA v2 / GQA 等数据子集
- 将 Caption、VQA、QA 等数据统一转换为指令对话格式
- 设计多任务采样策略，在 mini-LLaVA 上进行多任务 SFT
- 粗略评估各任务性能，观察任务间的互相影响

#### 第 11 周：多轮对话支持

- 学习 Qwen-VL / LLaVA 多轮对话数据格式
- 修改训练数据构造和推理脚本，支持多轮图文对话
- 加入简单的输出约束（如 JSON 格式回答），并总结多轮对话的实践经验

#### 第 12 周：整理 mini-LLaVA 工程 & 文档

- 清理工程目录，剔除无用脚本
- 为 mini-LLaVA 写较完整的 README：架构、训练步骤、推理示例
- 加日志系统、配置样例，完成一次端到端自测
- 输出：一个更「开源化」的 mini-LLaVA 子项目

---

### 第 13–16 周：指令对齐 & 安全策略（工程级）

#### 第 13 周：指令微调 & 格式规范化

- 学习 Chat 格式（system/user/assistant）和常见指令模板
- 将单轮、多轮数据统一为 Chat 格式（含 `<image>` 标记）
- 在 mini-LLaVA 上做一次 Chat 格式 SFT
- 对比不同对话格式的输出差异，并封装 Prompt 构造模块

#### 第 14 周：多轮对话稳定性 & 输出格式控制

- 压测长对话场景，分析遗忘图片信息、跑偏等问题
- 设计与实现上下文截断策略和 few-shot 输出模板（JSON、结构化回答）
- 封装输出解析模块（校验 JSON、失败重试等），构建多轮对话测试脚本

#### 第 15 周：安全策略 & 简单对齐思想

- 理解 RLHF/RLAIF/DPO 的基本概念与工程实现思路
- 设计并实现基础安全策略：输入过滤、输出过滤和越界场景测试
- 输出：安全策略说明文档 + 简单安全模块代码

#### 第 16 周：多任务训练调优 & 小结

- 调整多任务采样比例，比较单任务 vs 多任务训练的性能差异
- 整理一份「数据/任务混合经验」与调参心得
- 完整跑通一版：多任务 + Chat 格式 + 安全策略 的 mini-LLaVA
- 更新 README，做前 4 个月的阶段性复盘

---

### 第 17–20 周：推理加速 & 部署

#### 第 17 周：LLM 推理优化 & vLLM

- 安装 vLLM，跑通 Qwen / Llama 模型的服务化部署
- 对比 Transformers 原生推理与 vLLM 的性能（单请求、多并发）
- 理解 KV cache、prefill/decoding 等关键概念，写压测脚本并记录结果

#### 第 18 周：Vision Encoder TensorRT / OpenVINO

- 将 CLIP-ViT-B/16 导出为 ONNX，使用 ONNX Runtime 验证
- 转 TensorRT（FP16），测量加速效果，可选尝试 OpenVINO（CPU）
- 在 mini-LLaVA 推理脚本中替换 Vision 模块为 TRT/OV 推理

#### 第 19 周：端到端多模态服务化

- 使用 FastAPI 搭建 HTTP 服务，把 Vision Encoder + LLM 串成完整 pipeline
- 加入请求队列、简单限流和分阶段耗时日志
- 提供客户端脚本 / UI，实现可用多模态聊天/问答服务

#### 第 20 周：压测 & 性能优化

- 使用 `locust` / `wrk` 等工具对服务做压测
- 调整 batch size、max_new_tokens、并发度和 KV cache 等配置
- 输出一份性能报告（QPS、延迟、资源占用等），总结优化策略

---

### 第 21–24 周：代表性项目（例如：文档多模态问答系统）

#### 第 21 周：需求分析 & 系统设计

- 确定方向，如：文档/图表/UI 多模态问答系统
- 画架构图：上传 -> 解析 -> OCR -> 多模态模型 -> 回答
- 选型 OCR 模块（PaddleOCR 等）并调通，设计整体数据流和前端交互

#### 第 22 周：核心功能开发（端到端跑通）

- 实现 PDF 拆页（PDF -> 图片序列）与多页 OCR 解析
- 将 OCR 结果 + 图片喂入 mini-LLaVA / Qwen-VL / LLaVA 等模型进行问答
- 封装为后端 API，并打通前端：上传文档 -> 返回回答

#### 第 23 周：优化 & 工程加固

- 引入 Vision 加速（TensorRT/OV），优化多页文档整体耗时
- 针对常见场景（表格、标题等）做 Prompt 与逻辑优化
- 增加缓存、完善日志与错误处理，对系统做一轮压测

#### 第 24 周：包装 & 对外展示

- 完整整理项目 README：背景、功能、架构、技术亮点、使用说明
- 录屏/截屏作为展示素材，撰写技术文章（从 CV 到多模态工程实践）
- 检查代码风格、添加 LICENSE，评估开源方案
- 总结 6 个月的关键知识点、坑点与下一阶段规划

---

## 如何使用本仓库

### 适合人群

- 有一定 Python / 深度学习基础，希望系统入门 LLM / 多模态大模型的人  
- 对代表性多模态架构（CLIP / BLIP / LLaVA / Qwen-VL 等）感兴趣的工程实践者  
- 想从「会用模型」走向「能搭一个小型多模态系统并部署」的开发者

### 建议使用方法

1. 先阅读本 README 或 `docs/roadmap.md`，把握整体路线；
2. 选择你当前关心的阶段（例如：SFT、mini-LLaVA、部署）对应的 `weekXX_*/`；
3. 先看 `notes.md`，再跑 `code/` 中的脚本或 Notebook；
4. 遇到问题可参考 `docs/notes/` 中的总结与踩坑记录，或根据 README 中的参考链接查原论文/文档。

---

## 环境 & 依赖（示例）

> 各周目录可能有自己的 `requirements.txt` 或安装说明；  
> 这里给出一个全局基础环境，仅供参考。

- Python >= 3.10
- CUDA 对应版本的 PyTorch
- 常用库：
  - `transformers`, `datasets`, `accelerate`, `peft`, `bitsandbytes`
  - `torchvision`, `opencv-python`, `Pillow`
  - `gradio` 或 `fastapi`, `uvicorn`
  - `vllm`, `onnx`, `onnxruntime`, `tensorrt`（可选）
  - `pymupdf` / `pdf2image`, `paddleocr` 等（用于文档项目）

示例安装：

```bash
conda create -n mllm-study python=3.10
conda activate mllm-study

pip install -r requirements.txt
```

---

## 参考资料与致谢

计划中引用的大量开源项目、论文与教程会整理在：

- `docs/refs.md`：包括但不限于
  - LLaVA, MiniGPT-4, BLIP, BLIP-2, CLIP, Qwen/Qwen-VL 等模型
  - Hugging Face Transformers / PEFT / Datasets / TRL 文档
  - vLLM, TensorRT, OpenVINO 等部署与加速框架
  - 各类综述与教程文章

感谢这些开源工作为本仓库提供了丰富的实践素材与灵感。

---
## 抱团
如果你也在做类似的自学/实践，欢迎一起讨论、互相借鉴。

欢迎加入微信群一起学习一起成长

---

<div style="text-align: center;">
  <img src="docs/mllm.jpg" alt="MLLM交流学习群" style="zoom:20%;" />
</div>

