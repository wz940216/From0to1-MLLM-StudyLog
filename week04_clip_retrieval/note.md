# CLIP 入门 + 图文检索
pip install git+https://github.com/openai/CLIP.git

CLIP（Contrastive Language-Image Pretraining）使用了**对比学习（Contrastive Learning）**的方式来训练图像和文本的匹配关系，其核心思想是让正确的图像-文本对在特征空间中靠近，而错误的对则远离。

## 1. CLIP整体框架

CLIP 的目标是学习一个**共享语义空间**，使得匹配的图像和文本更接近，不匹配的更远。

给定一个 batch：

$$
{(I_i, T_i)}_{i=1}^N
$$

------

1.1 编码与归一化

- 图像编码：

  $$
  v_i = f_{\text{image}}(I_i)
  $$

- 文本编码：

  $$
  t_i = f_{\text{text}}(T_i)
  $$

- L2 归一化：

  $$
  v_i \leftarrow \frac{v_i}{|v_i|}, \quad t_i \leftarrow \frac{t_i}{|t_i|}
  $$

 归一化后：

$$
v_i^\top t_j = \cos(\theta_{i,j})
$$

L2 normalize 后：

$$
∥x∥=1∥y∥=1
$$

所有向量都被投到 **单位球面**

原本：

$$
dot(x,y)=∥x∥∥y∥cos⁡θ
$$

归一化后：

$$
dot(x,y)=cos⁡θ
$$

**只剩“方向”，没有“长度”**

如果不 normalize，会发生什么？

模型会“作弊”

```
把向量变得很大 → dot product 变大 → loss 变小
```

不需要学语义，只需要：

```
增大模长（norm explosion）
```

------

举个极端例子

```
x = [1, 0]
y = [1, 0]
dot = 1

x = [100, 0]
y = [100, 0]
dot = 10000   更大！
```

但语义完全一样

normalize 后的好处

✔ 防止“作弊”

→ 不能靠放大数值取胜

 强制学“方向”（语义）

→ 谁和谁“更接近”

✔ 数值稳定

→ logits 范围可控（配合 temperature）

------

## 2. 相似度矩阵（logits）

构造相似度矩阵：

$$
S_{i,j} = v_i^\top t_j
$$

加入 temperature（缩放）：

$$
\text{logits}*{i,j} = \frac{S*{i,j}}{\tau}
$$

------

## 3. 损失函数（InfoNCE / Cross Entropy）

CLIP 使用 **双向对比损失（bidirectional contrastive loss）**：

------

## 3.1 Image → Text（I→T）

对每一行做 softmax（图像作为 query）：

$$
\mathcal{L}*{i \to t} = - \frac{1}{N} \sum*{i=1}^{N}
\log \frac{\exp(S_{i,i}/\tau)}{\sum_{j=1}^{N} \exp(S_{i,j}/\tau)}
$$

含义：

- 固定图像 ( I_i )
- 在所有文本中找 ( T_i )

------

## 3.2 Text → Image（T→I）

对每一列做 softmax（文本作为 query）：

$$
\mathcal{L}*{t \to i} = - \frac{1}{N} \sum*{j=1}^{N}
\log \frac{\exp(S_{j,j}/\tau)}{\sum_{i=1}^{N} \exp(S_{i,j}/\tau)}
$$

含义：

- 固定文本 ( T_j )
- 在所有图像中找 ( I_j )

------

## 3.3 总损失

$$
\mathcal{L} = \frac{1}{2} \left( \mathcal{L}*{i \to t} + \mathcal{L}*{t \to i} \right)
$$

------

## 4. 与 Cross Entropy 的等价性

令：

$$
\text{logits} = S / \tau
$$

标签为：

$$
y_i = i
$$

那么：

```python
loss_i2t = F.cross_entropy(logits, labels)
loss_t2i = F.cross_entropy(logits.T, labels)
```

等价于上面的两个公式。

------

本质：

$$
\text{InfoNCE} \equiv \text{CrossEntropy}
$$

只是视角不同：

| 视角          | 含义                 |
| ------------- | -------------------- |
| Cross Entropy | 多分类问题           |
| InfoNCE       | 正样本 vs 多个负样本 |

------

## 5. 为什么需要双向损失？

5.1 logits 是方阵，但语义不同

相似度矩阵：

$$
S \in \mathbb{R}^{N \times N}
$$

- 第 i 行：图像 ( I_i ) vs 所有文本
- 第 j 列：文本 ( T_j ) vs 所有图像

------

5.2 Softmax 归一化不同

I→T（行 softmax）：

$$
\sum_{j} \exp(S_{i,j}/\tau)
$$

T→I（列 softmax）：

$$
\sum_{i} \exp(S_{i,j}/\tau)
$$

分母不同 ⇒ loss 不等价

------

## 6. 单向 loss 的问题

如果只用 I→T：

$$
\mathcal{L} = \mathcal{L}_{i \to t}
$$

可能出现：

- 多个图像匹配同一个文本
- embedding 空间塌缩（collapse）

------

## 7. 双向 loss 的作用

1. 强制一一匹配

目标变为：

$$
I_i \leftrightarrow T_i
$$

而不是：

$$
I_1, I_2 \rightarrow T_3
$$

------

2. 提供双重约束

- 行约束（image query）
- 列约束（text query）

------

  3. 更稳定的梯度

每个样本被优化两次：

- 一次作为 query
- 一次作为 target

------

  4. 提升对称性

学习到：

$$
\text{image} \leftrightarrow \text{text}
$$

而不是单向检索能力

------

## 8. 直觉总结

CLIP 的训练可以理解为：

> 在一个 batch 内做一个 **N-way matching problem**

------

单向：

> 每个图找一个正确文本

------

双向：

> 图和文本必须互相选择对方

（类似“双向匹配 / stable matching”）

------

## 9. 总结

CLIP 的损失函数可以概括为：

- 使用 **cosine similarity + temperature scaling**
- 构造 **N×N 相似度矩阵**
- 使用 **InfoNCE（= CrossEntropy）**
- 采用 **双向对比损失（I→T + T→I）**

------

核心优势：

1. 利用 batch 内样本作为负样本（高效）
2. 双向约束，防止 collapse
3. 学习统一的跨模态语义空间

# CLIP loss实验

```python
import torch
import torch.nn.functional as F


class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, image_features, text_features):
        """
        计算 CLIP 的对比损失

        参数：
        - image_features: 形状为 (N, D) 的图像特征
        - text_features: 形状为 (N, D) 的文本特征

        返回：
        - CLIP 损失
        """

        if image_features.dim() != 2 or text_features.dim() != 2:
            raise ValueError("image_features 和 text_features 必须是二维张量 (N, D)")
        if image_features.shape != text_features.shape:
            raise ValueError("image_features 和 text_features 必须具有相同的形状")

        # 归一化特征（L2 归一化到单位球面）
        image_features = F.normalize(image_features, p=2, dim=-1)  # (N, D)
        text_features = F.normalize(text_features, p=2, dim=-1)  # (N, D)

        # 计算相似度矩阵（余弦相似度）
        logits = (image_features @ text_features.T) / self.temperature  # (N, N)

        # 目标标签：对角线上的才是匹配的
        labels = torch.arange(logits.shape[0], device=logits.device)

        # 计算交叉熵损失（分别计算 I->T 和 T->I）
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # 这里 logits.shape = [16, 16] labels.shape = [16]
        # 可以这样理解：
        # logits[i]：第i个样本，对16个类别的“打分”
        # labels[i]：第i个样本的正确类别（0~15）
        # 也就是说：这是一个16分类问题，每个batch有16个样本
        # pytorch中的F.cross_entropy会先对logits[i]中的16个值做softmax，得到每个类别的概率分布
        # 再根据labels[i]指定的正确类别，计算交叉熵损失
        # 其实就是计算-log(p(correct_class_probability))，也就是正确类别的概率越大，损失越小

        return (loss_i2t + loss_t2i) / 2


if __name__ == '__main__':
    # 调试代码：使用随机图像/文本向量验证 CLIP 损失计算流程
    torch.manual_seed(123)

    batch_size = 16
    dim = 512
    image_features = torch.randn(batch_size, dim)
    text_features = torch.randn(batch_size, dim)

    clip_loss = CLIPLoss(temperature=0.07)
    loss_value = clip_loss(image_features, text_features)

    print(f"CLIP loss: {loss_value.item():.6f}")
 
```

# 动手写一个CLIP的前向推理
```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. 加载模型
model = CLIPModel.from_pretrained("models/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch16")

# 2. 准备数据
url = "week04_clip_retrieval/code/000000039769.jpg"
image = Image.open(url)

labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# 3. 处理输入
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# 4. 手动前向推理
with torch.no_grad():
    # 图像特征[1,512]
    image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
    
    # 文本特征[3,512]
    text_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

# 5. 归一化（CLIP 的关键步骤）
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 6. 手动计算相似度（cosine similarity）[1,3]
logits = image_features @ text_features.T   # (1, num_labels)

# CLIP 默认有一个 temperature scaling
logit_scale = model.logit_scale.exp()
logits = logits * logit_scale

# 7. softmax 得到概率
probs = logits.softmax(dim=1)

# 8. 选最大
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = labels[most_likely_idx]

print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")
```
