# SFT + LoRA / QLoRA 入门

## 理论基础

## 一、PEFT（Parameter-Efficient Fine-Tuning）基本原理

https://huggingface.co/docs/peft/index

https://hugging-face.cn/docs/peft/developer_guides/lora

LoRA、Prefix Tuning、P-Tuning、Prompt Tuning、IA³ 都属于“参数高效微调/PEFT”范畴

---

## 1. LoRA（Low-Rank Adaptation）

**核心思想**：  
不改动原模型权重，用低秩矩阵对某些线性层（如 W_q, W_k, W_v, W_o, FFN 等）做“增量更新”。

- 原线性层：  
  $y = W x$
- LoRA 之后：  
  $y = W x + BAx$  
  其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$，r 很小（低秩）。

**特点：**

- 修改位置：通常是 Transformer 中的线性层（attention/MLP）。
- 形式：给参数矩阵加一个“低秩旁路”。
- 训练参数量：远小于全量微调，但仍然在“层内部”增加参数。
- 优点：  
  - 对不同任务保存不同 LoRA 模块即可，方便多任务/多租户。  
  - 与原模型解耦，可以关闭/合并 LoRA。
- 缺点：需要改动模型结构（加入 LoRA 模块），不合并模型时推理稍有额外计算。

---

## 2. Prefix Tuning（前缀微调）

**核心思想**：  
在每一层的 self-attention 中，给每一层注入一段可学习的“虚拟 token 前缀”（prefix key/value），类似在每层加一串“任务向量”。

- 不改变原模型参数，只学习一组 prefix 参数：  
  - 对每一层：额外的 $K_{prefix}, V_{prefix}$ 序列  
  - 注意力时，真实输入序列会“看到”这段前缀。

**特点：**

- 修改位置：Transformer 各层的 attention 的 K/V。
- 形式：在**每层**拼接一段 learnable prefix（长度常数，比如 10～100 个虚拟 token）。
- 训练参数量：与层数 × prefix 长度 × 隐层维度相关，一般比 LoRA 还小。
- 优点：  
  - 不动原权重，完全在“输入上下文层面”调节行为。  
  - 对生成任务效果很好（最初工作在 PLM for generation）。
- 缺点：  
  - 实现要改 attention 的 KV 输入；  
  - prefix 太长会增加推理计算和显存。

---

## 3. P-Tuning / P-Tuning v2

### P-Tuning v1

**核心思想**：  
在输入层面加入一串可学习的“连续提示 embedding”（soft prompt），相当于在 embedding 层之前插入若干 learnable token。  
与 Prefix Tuning 的区别在于：  

- P-Tuning v1 通常只在**输入层**加 soft prompt；  
- Prefix Tuning 在**每一层** attention 中都有 prefix。

**特点：**

- 修改位置：嵌入层（embedding）的输入序列前后加 soft prompt。
- 形式：可学习的向量序列，而非自然语言 token。
- 训练参数量：仅与 prompt 长度 × embedding 维度有关，很少。
- 主要用于：早期在 GPT/BERT 上做分类、多任务。

### P-Tuning v2

P-Tuning v2 实际接近 **Prefix Tuning 的“分层版本+更通用”**：

- 在每一层注入 prompt（可理解为每层 prefix），不仅仅在输入层；
- 适用于 encoder-only、decoder-only、encoder-decoder 模型，任务范围更广；
- 强调“**prompt 代替全参数微调**”：只改 prompt，就能达到接近 full finetune 的效果。

简要对比：

- P-Tuning v1：只在 embedding 层前加 prompt；
- P-Tuning v2：层内注入 prompt（更像 prefix tuning，且更通用）。

---

## 4. Prompt Tuning（Soft Prompt Tuning，Google 提出的）

**核心思想**：  
和 P-Tuning v1 类似：在输入 token 前加一段可学习向量（soft prompts），原模型参数冻结，只训练这段向量。

**特点：**

- 修改位置：仅在 **输入 embedding** 层，添加固定长度的软提示向量。
- 形式：soft prompt = [p₁, p₂, …, p_m]，是 embedding 空间中的参数。
- 训练参数量：极小（只与 prompt 长度相关），可以十几、几十个 token。
- 优点：  
  - 实现最简单的 PEFT 方法之一；  
  - 对大模型时效果相对好（paper 发现模型越大，Prompt Tuning 越接近全参数微调效果）。
- 缺点：  
  - 对小模型可能效果不如全量微调；  
  - 表达能力受限于只在输入层改动。

> 总结：  
>
> - Prompt Tuning / P-Tuning v1：**只在 embedding 层前加软提示**；  
> - P-Tuning v2 / Prefix Tuning：**在多层 / 每层注入提示（prefix/prompt）**。

---

## 5. IA³（Infused Adapter by Inhibiting and Amplifying Inner Activations）

**核心思想**：  
给模型某些中间激活（如 attention 输出、FFN 中间层）加一维或低维的**可学习缩放因子**，通过逐元素缩放控制信息流（在不改动原权重的前提下“抑制/增强”激活）。

- 例如，在注意力中：  
  $\text{Attn}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d}) V$  
  IA³ 会插入对 V 或某些中间表示的缩放向量 
  $\lambda$：
  $\tilde{V} = \lambda \odot V$

**特点：**

- 修改位置：对 attention / FFN 的中间激活进行逐维缩放。
- 形式：额外学习一组缩放向量（一般是与通道数相同的 1D 向量）。
- 训练参数量：非常小，比 LoRA、Prefix Tuning 通常更小。
- 优点：  
  - 结构修改极小；  
  - 对极大模型的高效适配非常友好。  
- 缺点：  
  - 表达能力可能不如 LoRA 或多层 Prefix（因为只做缩放，空间较受限）。

---

## 6. 总体对比表（简化版）

| 方法          | 改动位置                        | 主要形式                          | 是否改原权重 | 典型参数量大小 | 适用场景/特点                        |
| ------------- | ------------------------------- | --------------------------------- | ------------ | -------------- | ------------------------------------ |
| LoRA          | 线性层（W_q, W_k, W_v, W_o 等） | 低秩矩阵 BA 叠加在权重上          | 否           | 中等           | 表达能力强，通用性好，多任务兼容性强 |
| Prefix Tuning | 各层 attention 的 K/V           | 每层前加一段 prefix KV            | 否           | 小             | 生成任务效果好，层内注入任务特征     |
| P-Tuning v1   | Embedding 输入                  | 输入序列前加 soft prompt 向量     | 否           | 很小           | 简单、轻量，多用于分类/问答等        |
| P-Tuning v2   | 多层（类似 prefix tuning）      | 各层注入 prompt，适配各种结构     | 否           | 小～中         | 更通用的“分层 prompt 化”             |
| Prompt Tuning | Embedding 输入                  | 输入前 soft prompt（Google 提出） | 否           | 很小           | 大模型上接近全微调效果，实践常用     |
| IA³           | Attention/FFN 中间激活          | 逐维缩放向量（scale）             | 否           | 极小           | 结构改动极小，极致参数高效           |

---

## 7. 怎样选择？

- **参数极少 + 极简实现**：首选 Prompt Tuning / IA³。  
- **想要更强表现、兼容多种任务**：LoRA 或 P-Tuning v2 / Prefix Tuning。  
- **已有产业实践**：目前很多开源/商用大模型微调框架以 LoRA 为主（或 QLoRA），soft prompt/prefix/IA³ 多作为补充。

## 8.区别与联系

P-tuning v1 和Prompt Tuning思想非常接近，都是固定模型参数，在embedding前增加一部分soft prompt进行训练。

P-tuning v2 和Prefix Tuning思想非常接近，都是固定模型参数，在transformer层内部注入一段可学习的向量。Prefix Tuning在decoder k v上进行注入，而P-tuning v2推广到更多区域，不局限与decoder的kv，也可在encoder或Q上进行注入。

**[Adapter-Tuning](https://zhida.zhihu.com/search?content_id=248624711&content_type=Article&match_order=1&q=Adapter-Tuning&zhida_source=entity)**：冻结原模型参数，在模型的层与层之间插入小型的 adapter 模块，仅对 adapter 模块进行训练。

## 二、各种LORA对比

---

## 一、LoRA 的核心思想

### 1. 为什么需要 LoRA？

大模型（例如几十亿参数）如果直接全参数微调：

- 显存开销大（需要存储参数 + 梯度 + 优化器状态）
- 训练慢，且每个下游任务需要保存一整份模型权重

LoRA 的目标是：

> **在冻结原始权重的前提下，只训练一个低秩矩阵，显著减少需要更新的参数量和显存。**

### 2. 基本公式

以某一层线性变换为例：

原来是：
$$
h = W x
$$

LoRA 把参数更新写成一个低秩分解：

$$
W' = W + \Delta W,\quad \Delta W = B A
$$

- $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$：原始权重（冻结）
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$
- $r \ll \min(d_{\text{out}}, d_{\text{in}})$：**秩（rank）**，通常 4/8/16 等

前向：
$$
h = W x + B (A x)
$$

训练时只更新 A、B；推理时可以合并：
$$
W_{\text{eff}} = W + B A
$$

特点：

- 参数量：原来是 $d_{\text{out}} \times d_{\text{in}}$，LoRA 只新增 $r (d_{\text{in}} + d_{\text{out}})$
- 显存：只需要存 LoRA 权重和对应梯度
- 多任务：每个任务存一份小 LoRA adapter 即可，共享底座模型 W

### 3. LoRA 通常加在哪

Transformer 中常见做法：

- 对 Attention 部分的 W_q, W_k, W_v, W_o 添加 LoRA
- 或者只对 W_q, W_v 添加（效果好且成本更低）
- 有些实现还会在 FFN 的 W_in、W_out 上加

---

## 二、QLoRA：量化 + LoRA

QLoRA = **4bit 量化的基座模型 + LoRA 微调**  
代表论文：*QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023)

### 1. 核心目标

- 不仅减少训练参数，还减少 **基座模型参数的显存占用**
- 让几十亿/百亿参数模型在单张 24/48GB 卡上也能微调

### 2. 关键技术点

1. **4-bit 量化基座模型（NF4）**
   - 用一种特定的 4-bit 格式（NF4：NormalFloat4），对权重做**非对称分组量化**
   - 量化后，基座模型权重以 4bit 存在显存中（或者 CPU + 分页），极大节省显存

2. **双重量化(“double quantization”)**

   - 先对权重分块计算 scale
   - 再对 scale 进行压缩，进一步节省内存

3. **冻结量化权重 + LoRA**

   - 量化后的权重 **完全冻结**，不参与更新
   - 在指定线性层上插入 LoRA（float16/bfloat16）
   - 只更新 LoRA 的 A, B

4. **Paged Optimizer（分页优化器）**

   - 把优化器状态按页管理，在需要时从 CPU / 磁盘分页进 GPU，缓解显存压力
   - 常见实现：`bitsandbytes` + `peft` + `transformers`

### 3. 前向与反向流程

- 前向：
  - 将 4bit 权重在计算时临时反量化到更高精度（如 float16）
  - 冻结的 W（量化存储） + 浮点 LoRA 再进行前向
- 反向：
  - 只对 LoRA 的参数计算梯度并更新
  - 量化权重不更新，因此不需要为其保存梯度和优化器状态

### 4. QLoRA 的典型效果

论文结论（以 65B LLaMA 为例）：

- 4-bit QLoRA 微调接近甚至逼近全精度全参数微调的性能
- 显存要求大幅降低（单卡 48GB 可以微调 65B）

实际中：

- 开源社区大量基于 LLaMA / Qwen / Mistral 的 QLoRA LoRA 模型
- 常配合 R=8/16，target_modules=“q_proj,v_proj” 或 “k_proj,v_proj,o_proj”等

---

## 三、常见 LoRA 变种

下面列出比较有代表性的 LoRA 类方法，并说明它们解决的问题、方法和现实使用情况。

---

### 1. AdaLoRA（Adaptive LoRA）

**问题**：普通 LoRA 的 rank r 是固定的，但不同层/不同训练阶段的重要性不同，固定 rank 不一定最优。

**核心思想**：  

> 自适应地为各层分配 LoRA 的秩，使得对重要层给更多秩，对不重要层给更少秩，从而在固定参数预算下提升性能。

**方法大意：**

- 初始时给每层一个较大的 rank
- 在训练过程中，根据某种重要性度量（例如 LoRA 权重的奇异值大小）对各层进行“剪枝压缩”：
  - 重要性小的方向（奇异值小）被裁剪掉
  - 重要性大的方向保留更多 rank
- 最终达到：总 LoRA 参数数大致不变，但分配更合理

**适用场景：**

- 显存/参数预算给定，希望最优利用
- 模型较大，层间重要性差异明显

---

### 2. LoRA-FA / LLaMA-Adapter / Bias-only / Prefix 等家族（结构不同）

并非严格“LoRA 变种”，但属于同一“参数高效微调（PEFT）”家族，对比了解一下：

1. **Bias-only Tuning（BitFit）**
   - 只训练所有层的 bias（偏置），其他参数冻结
   - 参数量极少，但能力有限
2. **Prefix / Prompt Tuning**
   - 在注意力的 key/value 前面拼接若干“虚拟 token 的隐层表示”
   - 只训练这些虚拟 token 的 embedding
3. **LLaMA-Adapter / LoRA-FA（某些实现名）**
   - 不仅对线性层增加低秩更新，也对特定结构（比如 attention 输出特征）插入轻量级 MLP/卷积等

理解要点：  
这些方法有的改“位置”（加在注意力 KV 上），有的改“结构”（加小 MLP），和 LoRA 都是 PEFT 家族成员，常做对比和组合。

---

### 3. DoRA（Decomposition of Rank-1 Updates / Weight Decomposition）

LoRA 的一个问题：  

- 原始 W 固定不变，所有更新在低秩 $\Delta W$ 上；  
- 有时会限制表示能力，而且在极低 rank 时会出现训练困难。

**DoRA 的核心思想**：  

> 把权重矩阵拆成 “尺度（magnitude）” 和 “方向（direction）”，只在“方向”上做低秩适配。

简单理解（其中一种表达方式）：

- 将原权重 $W$ 分解为：
  $$
  W = s \cdot \hat{W},\quad \|\hat{W}\|=1
  $$
  或近似为：若干 rank-1 或低秩的方向+尺度

- 冻结尺度 s，只对方向 $\hat{W}$ 做低秩更新（类似 LoRA）

- 保持原模型的“尺度结构”，减少训练不稳定

**实践感受（社区经验）：**

- 在极低 rank（例如 r=2、4）时，DoRA 可能比普通 LoRA 更稳定/效果更好
- 但复杂度也略高，且生态支持不如标准 LoRA/QLoRA 广泛

---

### 4. LoHa / LoKr：分解形式的变种

这些是“低秩 + 哈达玛积/克罗内克积”等结构化 LoRA，目标是“更高的表达力 / 参数利用率”。

#### 4.1 LoHa（Low-Rank Hadamard）

**核心思想**：  

> 用若干个低秩矩阵的哈达玛（逐元素）乘积来表示更新，使得参数数量类似，但表达更复杂。

形式上可以是：
$$
\Delta W = (B_1 A_1) \odot (B_2 A_2)
$$

- $\odot$：逐元素乘积
- 保持低秩子结构，但乘积带来更灵活的表示

在 Stable Diffusion 社区里有 LoRA/LoHa 等术语，LoHa 通常比标准 LoRA 在同等 rank 下更有表达力，但训练和推理的开销略大。

#### 4.2 LoKr（Low-Rank Kronecker）

**核心思想**：  

> 使用克罗内克积（Kronecker product）作为低秩结构，从而在空间上更有效地表示特定类型的线性变换。

形式可能类似：
$$
\Delta W = A \otimes B
$$

- $\otimes$：克罗内克积
- 利用矩阵的“结构化稀疏性”，达到比同参数量 LoRA 更强的表达

这两类在图像模型（如 diffusers 生态）比较常见，用于模型风格、概念的迁移与融合。

---

### 5. ReLoRA（Reset-then-LoRA / LoRA 重启训练）

ReLoRA 主要在大规模训练/继续预训练场景中被提及，思路大致是：

- 在训练中周期性地：
  1. 把 LoRA权重合并回基座权重 W
  2. 重置/重新初始化 LoRA
- 类似“多阶段重启 + 累积更新”的策略，有点像对大模型做渐进式训练、避免“LoRA 越堆越不稳定”。

更多是训练技巧，不是严格意义上的新的结构。

---

### 6. PiSSA / PiSSA-LoRA（基于 SVD 的参数初始化）

**问题**：  
LoRA 的 A, B 通常用随机初始化（如高斯/均匀分布），早期训练阶段效率不高。

**PiSSA（Principal Singular Subspace Adaptation）核心思想**：

> 用 SVD 分解原始权重 W 的主奇异向量来初始化 LoRA 的子空间，使 LoRA 一开始就对重要方向进行更新。

方法大意：

1. 对某层权重 W 做 SVD：
   $$
   W \approx U \Sigma V^T
   $$

2. 取前 r 个奇异向量构造 A, B 的初始值（或子空间）

3. 在此基础上微调

这样 LoRA 刚开始就“站在” W 的主方向上，而不是从随机方向开始学习，收敛更快，有时性能更好。

---

### 7. LLaMA-Adapter / LoRA 组合版本（如 LLaMA-Adapter v2 等）

这类工作经常把 LoRA 与其他 adapter 架构配合：

- 在注意力输入或输出拼接 adapter
- 加轻量 MLP
- 再在这些小模块内部用 LoRA 代替全参数训练

目的：同时利用  

- **结构层面**的额外表达力（Adapter）  
- **参数效率**（LoRA）

---

## 四、这些方法在实践中的选型建议

### 1. 最常用的组合

- **小显存 + 大模型微调**：  
  - 首选：**QLoRA（4bit） + LoRA**  
  - 工具：`transformers` + `peft` + `bitsandbytes`
- **有较多显存，希望最高效果**：
  - 尝试：全参数微调（全精度或 8bit+LoRA 混合）  
  - 或 QLoRA + 较高 rank + 更丰富 target_modules

### 2. LoRA rank 和 target_modules

- rank:
  - 4/8：轻量任务或资源有限
  - 16/32：更高任务难度，或希望更充分拟合
- target_modules:
  - 常见：`q_proj, v_proj` 或 `q_proj, k_proj, v_proj, o_proj`
  - 有的还在 `gate_proj`、`up_proj`、`down_proj` 上加

### 3. 什么时候考虑变种？

- **AdaLoRA**：
  - 预算固定，想尽量提升效果
  - 模型深、层间重要性不均
- **DoRA**：
  - rank 很小（≤4）时发现普通 LoRA 表现不稳定或不够好
- **LoHa / LoKr**：
  - 图像模型（如 Stable Diffusion）里，希望更细腻的风格/概念表达
- **PiSSA**：
  - 想加快收敛，或对初始阶段敏感（尤其是训练步数较少时）

---

## 五、简要对比总结

| 方法      | 主要思路                         | 优点                   | 缺点 / 代价                   |
| --------- | -------------------------------- | ---------------------- | ----------------------------- |
| LoRA      | 冻结 W，只训练低秩矩阵 BA        | 简单、稳定、生态成熟   | 需要全精度基座（若不用量化）  |
| QLoRA     | 4bit 量化 W + LoRA               | 显存极省，支持大模型   | 实现稍复杂，对框架依赖较大    |
| AdaLoRA   | 自适应分配 LoRA rank             | 在同参数预算下更高性能 | 训练逻辑复杂，主流实现较少    |
| DoRA      | 分离尺度/方向，只在方向上做 LoRA | 小 rank 时更稳定、更准 | 复杂度高，生态不如 LoRA/QLoRA |
| LoHa/LoKr | 哈达玛/克罗内克等结构化低秩更新  | 同参数下表达力更强     | 推理/训练开销略增，实现复杂   |
| ReLoRA    | 周期性合并 LoRA 并重启           | 对长时间训练更稳定     | 应用场景偏继续预训练，门槛高  |
| PiSSA     | 基于 SVD 的 LoRA 初始化          | 收敛更快，早期效果好   | 需要额外 SVD 计算，工程更复杂 |

## 代码实验

### mini sft

```python
# mini sft
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

# ----------------------------准备模型和数据----------------------------
# 1) 加载数据文件路径（train/validation）
data_files = {
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


```

### mini infer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# 基础模型路径（Qwen1.5 1.8B Chat）
base_model_name = "models/Qwen1.5-1.8B-Chat"
# LoRA 训练检查点目录（已训练好的 LoRA 权重）
lora_model_dir  = "week2/Qwen1.5-1.8B-Chat-lora/checkpoint-2000"   # 训练好的 LoRA 保存目录
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# 加载基础因果语言模型
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
# 加载 LoRA 权重到基础模型上
model = PeftModel.from_pretrained(base_model, lora_model_dir)
# 设置推理模式，并搬运到 GPU
model.eval().to("cuda")
# 准备输入指令文本
text = "给出保持健康的三个建议。"
prompt = f"指令: {text}\n输入: \n回复: "
# 将 prompt token 化为张量
inputs = tokenizer(prompt, return_tensors="pt")
# 用 LoRA 模型生成
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=250)
# 丢弃 prompt 部分，只取模型新生成的 token
gen_ids = outputs[0][inputs["input_ids"].shape[-1]:].detach().cpu().numpy()
# 将 token id 解码为文本（跳过特殊 token）
response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print("lora model response:")
print(response[0])
# 用基础模型生成（对照结果）
outputs = base_model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=250)
gen_ids = outputs[0][inputs["input_ids"].shape[-1]:].detach().cpu().numpy()
response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print("base model response:")
print(response[0])
```

### 微调后的效果

```shell
dataset:
"zh_instruction": "给出保持健康的三个建议。",
"zh_input": "",
"zh_output": "1.均衡饮食，确保多吃水果和蔬菜。\n2.定期锻炼，保持身体活跃和强壮。\n3.获得充足的睡眠并保持一致的睡眠时间表。",

lora model response:
健康饮食：多吃新鲜水果和蔬菜，避免糖分和饱和脂肪。
多运动：定期参加体育活动，例如跑步、骑自行车或游泳等。
睡眠充足：每晚尽量在同一时间上床睡觉并设定一个一致的作息时间表。

base model response:
保持健康的一些建议是吃营养均衡的食物，每天运动一小时，并保持充足的睡眠。此外，它还包括保持积极的心态，定期休息并避免压力。最后，定期进行体检也是保持健康的重要组成部分。这些都是一些有效的策略来保持身体健康。为了使这一切变得更容易，最好创建一个可追踪的日常生活计划，以确保你每天都采取行动来维护你的健康。
```



