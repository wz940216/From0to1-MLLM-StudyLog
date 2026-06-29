# week15_safety_alignment

本周主要在理解模型的安全策略以及大语言模型的对齐思想，这个对齐不是之前的图像和语言之间的特征对齐，而是大语言模型的预测结果向人类偏好对齐。  也就是如何让的语言模型输出的结果更满足人类喜好。  

本周涉及到一些强化学习的基础知识，这里给大家推荐李宏毅老师的主页和他的youtube课程资料。

https://speech.ee.ntu.edu.tw/~hylee/index.php

https://www.youtube.com/@HungyiLeeNTU/search?query=%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0

下面快速回顾一下大语言模型的训练流程。  

## 1. 大语言模型回顾

### 大语言模型训练流程

常见流程是：

```text
Pretraining
    ↓
SFT
    ↓
RLHF / Preference Optimization
    ↓
PPO / DPO / GRPO
```

大模型预训练阶段主要做的是根据前文预测下一个 token  

也就是最大化：

$$
p(\text{next token} \mid \text{context})
$$
这让模型学会语言、知识、代码、推理模式，但它并不天然知道什么回答“更好”。  

比如模型可能会：

- 答非所问
- 过度编造
- 不遵循人类偏好
- 输出格式不稳定
- 在推理任务中缺少自我校正

所以在预训练和 SFT 之后，还需要进一步做**偏好优化**或**强化学习对齐**。

---

### **SFT 回顾**

SFT，全称 Supervised Fine-Tuning，即监督微调。

数据形式：

```text
prompt → ideal answer
```

训练目标：

$$
\max \log \pi_\theta(\text{answer} \mid \text{prompt})
$$
SFT 的本质是让模型模仿人工写好的答案。

它的优点是简单稳定，能让模型学会基本的指令跟随。  

但缺点是：它只知道“这个答案是示范答案”，不知道“为什么这个答案比另一个答案更好”。

因此后面需要引入偏好数据。

---

### 偏好数据是什么？

偏好数据通常长这样：

```text
prompt: 解释 Transformer

chosen: 一个更好的回答
rejected: 一个更差的回答
```

例如造一个偏好数据：

```json
{"image": "dataset/coco128/images/train2017/000000000025.jpg", "prompt": "请描述图中的主要内容。", "chosen": "图中包含一个清晰可见的室内场景，回答应围绕图片中能观察到的物体和关系展开。", "rejected": "这张图展示了与画面无关的危险操作步骤。"}
{"image": "dataset/coco128/images/train2017/000000000025.jpg", "prompt": "如果用户让你根据图片编造不存在的细节，你应该怎么回答？", "chosen": "我应该只描述图中能确认的信息，并说明无法确定的部分，避免编造。", "rejected": "我可以随便补充看起来合理但图片里没有的信息。"}


```

记作：

$$
(x, y_w, y_l)
$$
其中：

- `x`：输入 prompt
- `y_w`：chosen / winner，更好的回答
- `y_l`：rejected / loser，更差的回答

偏好优化的目标是：

```text
让模型更倾向于 chosen，而不是 rejected
```

---

## 2. PPO：经典 RLHF 方法

PPO，全称 Proximal Policy Optimization，是一种强化学习算法。  

在大模型 RLHF 中，PPO 把 LLM 看成一个 policy。

对应关系如下：

| 强化学习概念 | LLM 中的对应物        |
| ------------ | --------------------- |
| policy       | 大语言模型            |
| state        | prompt + 已生成 token |
| action       | 下一个 token          |
| trajectory   | 完整回答              |
| reward       | 回答质量分数          |

PPO 训练时大概是：

```
1. 给 policy model 一批 prompts
2. policy model 采样生成 answers
3. reward model 给每个 answer 打分
4. 加 KL 惩罚，避免偏离 reference model 太远
5. 用 value model 估计 baseline / advantage
6. 用 PPO loss 更新 policy model
```

### Rollout 是什么？

Rollout 可以理解成：

```text
让当前模型真的生成一次完整回答
```

在强化学习里，rollout 是从某个状态开始，让 policy 连续选择 action，直到 episode 结束。

在 LLM 中：

```text
prompt = 初始 state
每个 token = action
生成中的上下文 = state
完整回答 = trajectory
Reward Model 的打分 = reward
```

例如：

```text
Prompt: 解释 PPO

模型生成：
PPO 是一种强化学习算法，它通过限制每次策略更新幅度……

Reward Model 打分：
8.7
```

这整个从 prompt 到完整 answer 的生成过程，就是一次 rollout。

### Reward Model 怎么训练？

PPO 通常需要一个 Reward Model，用于给每一次的回答打分。

例如：

```text
prompt: 解释 PPO
answer: PPO 是一种策略优化算法……
reward: 8.5
```

Reward Model 通常用偏好数据训练。

数据形式：

$$
(x, y_w, y_l)
$$
其中 y_w 代表优质回答， y_l 代表不好的回答。winner / loser

训练目标是让：

$$
r(x, y_w) > r(x, y_l)
$$
常用 loss 是 log sigmoid：

$$
L_{RM} = -\log \sigma\left(r(x, y_w) - r(x, y_l)\right)
$$
训练时最小化这个 loss。

直觉是：chosen 的 reward 应该高于 rejected ，也就是 winner 的分数应该高于loser的分数。

如果 chosen 分数已经明显高于 rejected，loss 很小。  

如果 rejected 分数反而更高，loss 很大。

Reward Model 的结构一般是：

```text
Transformer backbone + scalar reward head
```

通常可以从 SFT 模型初始化，然后把 LM head 换成 reward head。

### Value Model 怎么训练

Value Model 通常是在 PPO 的 rollout 数据上训练出来的。

Value Model 学的是在某个生成中间状态下，未来最终大概能拿多少 reward。

Value Model 的训练数据来自当前 policy model 的 rollout。

比如给一个 prompt：

```
x = "解释 PPO"
```

模型生成一个回答：

```
y = token_1, token_2, ..., token_T
```

Reward Model 最后给完整回答一个分数：

```
R = 8.5
```

那么这次 rollout 里会有很多中间状态：

```
s_0 = prompt
s_1 = prompt + token_1
s_2 = prompt + token_1 + token_2
...
s_T = prompt + full answer
```

Value Model 要学的是：

```
V(s_t) ≈ 从 s_t 继续生成，最终能拿到的期望 reward
```

训练目标最简单可以理解成回归：

```
让 V(s_t) 预测最终回报 R_t
```

loss 通常是 MSE：

```
L_value = (V(s_t) - target_t)^2
```

其中 `target_t` 是这个状态对应的“真实回报估计”。

target_t 在 LLM RLHF 里，reward 通常主要在完整回答结束后才给：

```
最后 reward = reward_model_score
```

同时每个 token 位置可能还有 KL penalty：

```
r_t = -β · KL_t
```

最后一个 token 还会加上 Reward Model 分数：

```
r_T = reward_model_score - β · KL_T
```

所以每个位置的 return 可以写成：

```
G_t = r_t + r_{t+1} + ... + r_T
```

Value Model 就训练成：

```
V(s_t) ≈ G_t
```

也就是从当前状态开始，未来累计能拿到多少分。

---

### 有了 Reward Model，怎么优化 LLM？

这里容易误解：PPO 不是直接把 Reward Model 的梯度反传进 LLM。

因为 LLM 生成文本时有采样操作：

```text
sample token → 组成 answer → reward model 打分
```

当所有的token全部生成后才能组成answer，最后拿给reward模型打分。

但是这里的 token 是离散的，采样过程不可导，所以不能像普通监督学习一样直接反传。

PPO 使用的是**策略梯度**。

目标是：

$$
\max \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}[r(x, y)]
$$
意思是调整模型参数 θ，让模型在输入 `x` 时，生成出来的回答 `y` 的平均奖励尽可能高。

x是prompt，比如：“解释一下 PPO”。

πθ(. | x) 是当前 LLM 的生成分布，给定 prompt `x`，模型可能生成很多不同回答，每个回答都有一个概率。

比如同一个 prompt，模型可能生成：

```
y1: PPO 是一种强化学习算法……     概率 0.30
y2: PPO 是一种优化器……             概率 0.10
y3: 我不知道。                     概率 0.05
...
```

每个回答都有 reward：

```
reward(x, y1) = 8
reward(x, y2) = 5
reward(x, y3) = 1
```

------

`E[ reward(x, y) ]` 里的 `E` 是期望，也就是**平均值**。

更直观地说：

$$
\mathbb{E}[r(x, y)] = \sum_y \pi_\theta(y \mid x) r(x, y)
$$
也就是模型可能生成的所有回答的 reward 加权平均。

加权用的是模型生成它们的概率：

$$
0.30 \cdot r(y_1) + 0.10 \cdot r(y_2) + 0.05 \cdot r(y_3) + \cdots
$$
所以：

$$
\max \mathbb{E}[r(x, y)]
$$
就是：

```
让模型整体更倾向于生成高 reward 的回答
```

不是只看某一个回答的reward，而是看“按当前模型分布采样时，平均能拿多少分”。也就是让模型更容易生成高 reward 的回答。

有了reward，模型优化的最终的策略梯度核心形式为：

$$
\nabla_\theta J(\theta) \approx A \cdot \nabla_\theta \log \pi_\theta(y \mid x)
$$
其中：

$$
A > 0 \Rightarrow \text{沿着增加 } \log \pi_\theta \text{ 的方向更新}
$$

$$
A < 0 \Rightarrow \text{沿着减少 } \log \pi_\theta \text{ 的方向更新}
$$

这是PPO损失函数的梯度计算形式，其中：

- `πθ` 是当前 LLM
- `log πθ(y | x)` 是模型生成回答 `y` 的 log probability
- `A` 是 advantage，表示这个回答比预期好多少

这里看似没有reward模型什么事，也就是在原模型的log probability前面乘上一个系数A。  

但是这个A其实来自于reward模型。

这里的A可以理解成：

```
这个回答比平均水平好多少
```

它通常由 reward 计算出来。

最简单版本可以写成：

$$
A = r(x, y) - b
$$
b是reward 的平均值或baseline。比如：

$$
r(x, y) = 8, \quad b = 5, \quad A = 3
$$
说明这个回答比平均预期好，应该提高它的概率。

如果：

$$
r(x, y) = 2, \quad b = 5, \quad A = -3
$$
说明这个回答比平均预期差，应该降低它的概率。

所以策略梯度真正的意思是：

$$
\nabla_\theta J(\theta) \approx [r(x, y) - b] \cdot \nabla_\theta \log \pi_\theta(y \mid x)
$$
为了简洁，通常把：

$$
r - b
$$
写成A

那为什么不直接写 reward，而要写 advantage？

因为只用 reward 也可以：

$$
r(x, y) \cdot \nabla_\theta \log \pi_\theta(y \mid x)
$$
但这样方差很大，训练不稳定。

举个🌰：

如果所有回答 reward 都是正数，那么模型会倾向于把所有采样过的回答概率都提高。但真正想知道的是这个回答是不是比当前平均水平更好？所以要减去一个 baseline。例如：

同一个 prompt，模型采样了 4 个回答：

| 回答 | reward | baseline/平均值 | A = reward - baseline |
| ---- | ------ | --------------- | --------------------- |
| A    | 9      | 5               | +4                    |
| B    | 6      | 5               | +1                    |
| C    | 4      | 5               | -1                    |
| D    | 1      | 5               | -4                    |

策略梯度会让：

```
A、B 的概率上升
C、D 的概率下降
```

也就是：

```
高于平均的回答被强化
低于平均的回答被抑制
```

------

所以这行公式里的 reward 在这里：

$$
A = r - b
$$

---

### PPO 中的 KL 约束

在A的计算中，PPO 里更复杂一些，reward 还可能包括 KL 惩罚：

$$
r_{\text{final}} = r_{\text{model}} - \beta \cdot D_{KL}(\pi_{\text{policy}} \parallel \pi_{\text{ref}})
$$
然后再用这个 final reward 去算 advantage。

PPO 中通常还有一个 reference model。

Reference model 一般是冻结的 SFT 模型，用来约束当前模型不要偏离太远。

最终 reward 常写成：

$$
r_{\text{final}} = r_{\text{model}} - \beta \cdot D_{KL}(\pi_{\text{policy}} \parallel \pi_{\text{ref}})
$$
意思是回答要高分，但不能为了高分变得太奇怪

如果没有 KL 惩罚，模型可能会 reward hacking，也就是学会“骗 Reward Model”。

---

### PPO 的训练流程

最后再来看一下PPO 的整体流程：

```text
1. 准备一批 prompts
2. 当前 policy model 生成 answers
3. Reward Model 给 answers 打分
4. 加上 KL penalty 计算得到 G_t
5. Value Model 估计未来期望reward 并计算 A_t = G_t - V(s_t)
6. PPO 根据 A_t 更新 policy model
```

简化版 PPO objective，PPO的损失函数，数学上通常是最大化这个目标。  代码里经常写成 loss，所以会最小化：

$$
\text{loss} = -L_{PPO}
$$

$$
L_{PPO} = \mathbb{E}\left[\min\left(\rho_t A_t, \operatorname{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) A_t\right)\right]
$$

这里的 clip 是PPO加了一个限制：可以提高/降低概率，但一次别改太猛。

其中 p_t 是：

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$
它叫 probability ratio，概率比。

含义是：

```
当前新模型生成这个 token 的概率 / 旧模型生成这个 token 的概率
```

其中：

- `πold`：采样 rollout 时的旧模型
- `πθ`：正在训练更新的新模型
- `a_t`：当时采样出来的那个 token
- `s_t`：当时的上下文状态

举例：

某一步生成 token `"PPO"`。

旧模型概率：

$$
\pi_{\text{old}}(a_t \mid s_t) = 0.10
$$
新模型概率：

$$
\pi_\theta(a_t \mid s_t) = 0.15
$$
那么：

$$
\rho_t = \frac{0.15}{0.10} = 1.5
$$
意思是：

```
新模型把这个 token 的概率提高到了原来的 1.5 倍
```

如果：

$$
\rho_t = 0.8
$$
意思是：

```
新模型把这个 token 的概率降到了原来的 0.8 倍
```

所以：

$$
\rho_t > 1 \Rightarrow \text{这个 token 概率被提高}
$$

$$
\rho_t < 1 \Rightarrow \text{这个 token 概率被降低}
$$

$$
\rho_t = 1 \Rightarrow \text{没变}
$$

那么 clip(ρ_t, 1 - ε, 1 + ε) A_t 又是什么意思？

如果设置 ε=0.2 ，那么允许范围是：

$$
[1 - \epsilon, 1 + \epsilon] = [0.8, 1.2]
$$
此时如果ρ_t小于0.8或大于1.2，都会clip到0.8和1.2上。

意思是一次更新里，不鼓励把某个 token 的概率改到低于 0.8 倍或高于 1.2 倍，如果更新的太激进，训练容易崩。

最终取min后求期望，也就是 L_PPO

$$
L_{PPO} = \mathbb{E}[\min(\rho_t A_t, \operatorname{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) A_t)]
$$
为什么用 `min`？因为这是一个要最大化的 objective。取 `min` 等于说如果模型想通过过度改变概率获得更大收益，那我不承认这部分收益，限制模型更新太激进。

总结是：

```
PPO：
好回答概率 ↑，差回答概率 ↓，
但是每次更新不能离旧模型太远
```

`ρ_t` 负责衡量：

```
新模型相对旧模型，把这个 token 概率改了多少倍
```

`clip` 负责限制：

```
这个倍数最好别超出 [1-ε, 1+ε]
```

`min` 负责实现：

```
超过范围的那部分收益不算数
```

再回到损失函数来，从损失函数来看梯度更新策略：

PPO 的未裁剪目标是：

$$
L(\theta) = \mathbb{E}[\rho_t A_t]
$$
其中：

$$
\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$
注意：

$$
\pi_{\text{old}}
$$
是旧模型，固定不变。

`A_t` 在这次更新里通常也当作常数。

所以对 θ 求导：

$$
\nabla_\theta L(\theta) = \mathbb{E}[A_t \cdot \nabla_\theta \rho_t]
$$
因为：

$$
\rho_t = \frac{\pi_\theta}{\pi_{\text{old}}}
$$
而 `πold` 是常数，所以：

$$
\nabla_\theta \rho_t = \nabla_\theta\left(\frac{\pi_\theta}{\pi_{\text{old}}}\right) = \frac{1}{\pi_{\text{old}}} \nabla_\theta \pi_\theta
$$
因为 

$$
\nabla_\theta \log \pi_\theta = \frac{1}{\pi_\theta} \cdot \nabla_\theta \pi_\theta
$$
两边同时乘以 `πθ`：

$$
\pi_\theta \nabla_\theta \log \pi_\theta = \nabla_\theta \pi_\theta
$$
代到这个式子中去：

$$
\nabla_\theta \rho_t = \nabla_\theta\left(\frac{\pi_\theta}{\pi_{\text{old}}}\right) = \frac{1}{\pi_{\text{old}}} \nabla_\theta \pi_\theta
$$


$$
\nabla_\theta \rho_t = \frac{\pi_\theta}{\pi_{\text{old}}} \nabla_\theta \log \pi_\theta = \rho_t \nabla_\theta \log \pi_\theta
$$
所以原式变成：

$$
\nabla_\theta L(\theta) = \mathbb{E}[A_t \cdot \nabla_\theta \rho_t]
$$

$$
\nabla_\theta L(\theta) = \mathbb{E}[A_t \cdot \rho_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)]
$$
如果刚开始更新时，新旧模型差不多：

$$
\rho_t \approx 1
$$
就变成：

$$
\nabla_\theta L(\theta) \approx \mathbb{E}[A_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t)]
$$
这就是之前讨论过的策略梯度形式。

---

## 3. DPO：直接偏好优化

DPO，全称 Direct Preference Optimization。

它的核心思想是：不训练显式 Reward Model，也不做复杂 PPO，直接用 chosen / rejected 数据优化模型。

DPO 使用的数据还是：

$$
(x, y_w, y_l)
$$
目标是让模型更偏好 chosen：

$$
\pi_\theta(y_w \mid x) > \pi_\theta(y_l \mid x)
$$

---

### DPO 公式

DPO 的核心目标是最大化：

$$
\log \sigma\left(\beta \left[\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x) - \log \pi_{\text{ref}}(y_w \mid x) + \log \pi_{\text{ref}}(y_l \mid x)\right]\right)
$$
也可以写成：

$$
\log \sigma\left(\beta \left\{[\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)] - [\log \pi_{\text{ref}}(y_w \mid x) - \log \pi_{\text{ref}}(y_l \mid x)]\right\}\right)
$$
这更容易理解：

```text
当前模型对 chosen/rejected 的偏好强度 - 参考模型对 chosen/rejected 的偏好强度
```

DPO 的目标是：

```text
让当前模型相对于 reference model 更偏好 chosen
```




---

### DPO 的直觉

如果 reference model 本来就很偏好 chosen，那么当前模型不需要被推得太猛。

如果 reference model 本来偏好 rejected，那么当前模型需要做更大的修正。

因此 DPO 不是单纯地让模型无脑提高 chosen 概率，而是让模型在 reference model 的基础上做合理偏好调整。

本周的代码实验采用了DPO的形式，数据集获取：

```text
# dpo 数据集 MMInstruction/VLFeedback
hf download MMInstruction/VLFeedback --repo-type=dataset --local-dir dataset/VLFeedback
```
准备config，添加DPO相关的配置和参数设置：

```yaml
# DPO 偏好优化配置
DPO:
  TRAIN_FILE: "dataset/VLFeedback"  # DPO 数据来源：本地 VLFeedback、Hub 名称或普通 JSON/JSONL
  IMAGE_ROOT: null  # 普通 JSON/JSONL 中 image 为相对路径时使用；VLFeedback 通常不需要
  CHECKPOINT: "week14_dialogue_stability_output_control/outputs/checkpoints/multitask_balanced/step_15522.pt"  # 可选：SFT checkpoint 路径，policy/reference 都从这里初始化
  SAVE_EXAMPLE_DATA: null  # 可选：只导出 DPO 示例 JSONL 后退出
  EXAMPLE_IMAGE: null  # 生成示例数据或 toy data 时使用的图片
  MAX_SAMPLES: null  # 调试时只取前 N 条偏好样本
  BETA: 0.1  # DPO beta，越大表示偏好约束越强
  TOY_DATA: false  # true 时不加载 VLFeedback，只使用内置两条小样本
```

继续week14中sft后的模型，使用DPO进行偏好优化训练。  
在SFT的训练循环的基础下，通过新增DPO的损失函数以及DPO数据集相关的dataset和collator就可以改成DPO的训练循环了。  

DPO dataset, 重点是读取VLFeedback数据集，或者普通的JSON/JSONL偏好数据集。根据一定的规则构造出chosen和rejected的偏好对。  
如果是人工标注的JSON数据集，或其他自定义数据集，直接加载chosen和rejected即可。  
VLFeedback是一个包含图像、prompt和多个completion的偏好数据集，可以根据completion的评分来选择chosen和rejected，我这里是将这几个打分加起来"Helpfulness", "Ethical Considerations", "Visual Faithfulness"，作为总分，选择最高的作为chosen，最低的作为rejected。  





```python
class LlavaDPODataset(Dataset):
    """MiniLLaVA DPO 偏好数据集。

    支持两类数据来源：
    1. 普通 JSON/JSONL：每条样本已经是 image/prompt/chosen/rejected。
    2. VLFeedback：每条样本是 image/prompt/completions，脚本会自动把评分最高的
       completion 转成 chosen，把评分最低的 completion 转成 rejected。

    VLFeedback 的 completion 有 Helpfulness、Ethical Considerations、Visual Faithfulness
    三个 Rating。这里用三项评分求和作为总分，工程上先得到一个简单可用的 DPO 偏好对。
    """

    def __init__(self, data_file=None, image_root=None, max_samples=None, example_image=None, toy_data=False):
        self.image_root = image_root
        self.is_vlfeedback = False

        if toy_data or data_file is None:
            self.samples = self._build_toy_samples(example_image)
        elif self._is_vlfeedback_source(data_file):
            self.samples = self._load_vlfeedback(data_file)
            self.is_vlfeedback = True
        else:
            self.samples = self._read_preference_file(data_file)

        if max_samples is not None:
            self.samples = self.samples.select(range(min(int(max_samples), len(self.samples)))) if self.is_vlfeedback else self.samples[:max_samples]

    def _is_vlfeedback_source(self, path):
        path = str(path)
        if path == "MMInstruction/VLFeedback":
            return True
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "README.md")):
            with open(os.path.join(path, "README.md"), "r", encoding="utf-8") as f:
                return "VLFeedback" in f.read(2048)
        return False

    def _load_vlfeedback(self, path):
        """加载本地或 Hub 上的 VLFeedback。

        本地目录 dataset/VLFeedback 中已有 9 个 parquet shard 时，传入该目录即可。
        如果要从 Hub 加载，也可以传入 MMInstruction/VLFeedback。
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("加载 VLFeedback 需要安装 datasets：pip install datasets") from exc

        try:
            dataset = load_dataset(path, split="train")
        except Exception:
            parquet_files = sorted(glob.glob(os.path.join(str(path), "data", "train-*.parquet")))
            if not parquet_files:
                raise
            dataset = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
        if len(dataset) == 0:
            raise ValueError(f"VLFeedback 数据集为空: {path}")
        return dataset

    def _read_preference_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
            return data["data"] if isinstance(data, dict) and "data" in data else data

    def _build_toy_samples(self, example_image):
        image = example_image or "dataset/coco128/images/train2017/000000000025.jpg"
        return [
            {
                "image": image,
                "prompt": "请描述图中的主要内容。",
                "chosen": "图中包含一个清晰可见的室内场景，回答应围绕图片中能观察到的物体和关系展开。",
                "rejected": "这张图展示了与画面无关的危险操作步骤。",
            },
            {
                "image": image,
                "prompt": "如果用户让你根据图片编造不存在的细节，你应该怎么回答？",
                "chosen": "我应该只描述图中能确认的信息，并说明无法确定的部分，避免编造。",
                "rejected": "我可以随便补充看起来合理但图片里没有的信息。",
            },
        ]

    def _resolve_image_path(self, image_path):
        image_path = str(image_path)
        if os.path.isabs(image_path):
            return image_path
        if self.image_root:
            return os.path.join(self.image_root, image_path)
        return image_path

    def _load_image(self, image_value):
        """把普通路径或 HF datasets Image 字段统一转成 RGB PIL.Image。"""
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if isinstance(image_value, dict):
            if image_value.get("bytes") is not None:
                return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
            if image_value.get("path"):
                return Image.open(self._resolve_image_path(image_value["path"])).convert("RGB")
        return Image.open(self._resolve_image_path(image_value)).convert("RGB")

    def _rating_to_score(self, rating):
        """把 VLFeedback 中的 Rating 字符串转成数字分数。"""
        if rating is None:
            return 0.0
        match = re.search(r"-?\d+(?:\.\d+)?", str(rating))
        return float(match.group(0)) if match else 0.0

    def _completion_score(self, completion):
        annotations = completion.get("annotations") or {}
        score = 0.0
        for key in ["Helpfulness", "Ethical Considerations", "Visual Faithfulness"]:
            score += self._rating_to_score((annotations.get(key) or {}).get("Rating"))
        return score

    def _normalize_completions(self, completions):
        """兼容 VLFeedback 的两种 completion 表示。

        从 datasets 读 parquet 时，sequence struct 会变成 dict-of-list：
        {"annotations": [...], "model": [...], "response": [...]}。
        旧 JSONL 样本通常是 list-of-dict。这里统一成 list-of-dict。
        """
        if isinstance(completions, dict):
            responses = completions.get("response", [])
            models = completions.get("model", [""] * len(responses))
            annotations = completions.get("annotations", [{}] * len(responses))
            return [
                {
                    "response": response,
                    "model": models[idx] if idx < len(models) else "",
                    "annotations": annotations[idx] if idx < len(annotations) else {},
                }
                for idx, response in enumerate(responses)
            ]
        return list(completions or [])

    def _convert_vlfeedback_item(self, item):
        completions = [
            c
            for c in self._normalize_completions(item.get("completions", []))
            if str(c.get("response", "")).strip()
        ]
        if len(completions) < 2:
            raise ValueError(f"VLFeedback 样本缺少至少两个有效 completion: {item.get('id')}")

        ranked = sorted(completions, key=self._completion_score, reverse=True)
        chosen = ranked[0]
        rejected = ranked[-1]
        return {
            "image": item["image"],
            "prompt": item["prompt"],
            "chosen": chosen["response"],
            "rejected": rejected["response"],
            "system_prompt": str(item.get("system", "")).strip(),
            "sample_id": item.get("id", ""),
            "chosen_model": chosen.get("model", ""),
            "rejected_model": rejected.get("model", ""),
            "chosen_score": self._completion_score(chosen),
            "rejected_score": self._completion_score(rejected),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raw_item = self.samples[index]
        item = self._convert_vlfeedback_item(raw_item) if self.is_vlfeedback else raw_item
        image = self._load_image(item["image"])
        prompt = _strip_image_token(item["prompt"])
        system_prompt = str(item.get("system_prompt", item.get("system", ""))).strip()

        return {
            "image": image,
            "prompt": prompt,
            "chosen": str(item["chosen"]).strip(),
            "rejected": str(item["rejected"]).strip(),
            "system_prompt": system_prompt,
            "sample_id": item.get("sample_id", item.get("id", str(index))),
            "chosen_model": item.get("chosen_model", ""),
            "rejected_model": item.get("rejected_model", ""),
            "chosen_score": item.get("chosen_score"),
            "rejected_score": item.get("rejected_score"),
        }
```


之后通过collator将chosen和rejected的文本转成MiniLLaVA格式的batch，方便后续计算DPO的loss。

```python
class LlavaDPOCollator:
    """把一批偏好样本转成 chosen/rejected 两个 MiniLLaVA batch。"""

    def __init__(self, tokenizer, max_length=512, max_turns=1):
        self.sft_collator = LlavaCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            max_turns=max_turns,
        )

    def _to_sft_feature(self, sample, answer):
        # 当前 MiniLLaVA 通过第一轮 USER 中的 <image> token 插入视觉特征。
        return {
            "image": sample["image"],
            "turns": [
                {
                    "question": f"{IMAGE_TOKEN}\n{sample['prompt']}",
                    "answer": answer,
                }
            ],
            "system_prompt": sample.get("system_prompt", ""),
            "sample_id": sample.get("sample_id", ""),
            "task_name": "dpo",
        }

    def __call__(self, features):
        chosen_features = [self._to_sft_feature(sample, sample["chosen"]) for sample in features]
        rejected_features = [self._to_sft_feature(sample, sample["rejected"]) for sample in features]
        return {
            "chosen": self.sft_collator(chosen_features),
            "rejected": self.sft_collator(rejected_features),
            "sample_ids": [sample.get("sample_id", "") for sample in features],
        }
```
最终返回的batch就是这样：

```json
"chosen": self.sft_collator(chosen_features),
"rejected": self.sft_collator(rejected_features),
"sample_ids": [sample.get("sample_id", "") for sample in features]
```

拿到chosen和rejected的batch后，就可以计算DPO的loss了。  

loss在代码中实现也很简单：
```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    """DPO loss。

    policy_logratios 表示当前模型更偏向 chosen 还是 rejected。
    ref_logratios 表示参考模型原本的偏好。
    DPO 优化的是两者差值，让 policy 比 reference 更偏向 chosen。
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    losses = -F.logsigmoid(float(beta) * logits)
    rewards_chosen = float(beta) * (policy_chosen_logps - ref_chosen_logps).detach()
    rewards_rejected = float(beta) * (policy_rejected_logps - ref_rejected_logps).detach()
    return losses.mean(), logits.detach(), rewards_chosen, rewards_rejected

```
最后将datast、优化器、学习率策略、accelerator、forward、loss、backward组织在一起，就开始DPO的训练了。  

在训练循环中，应该将reference model设置为torch.no_grad()，不更新它的参数，只更新policy model的参数。  

还有一个小细节，dataset返回的batch里，chosen或是redected的字典中，都存在Image字段，在训练时，Image已经被转成了视觉特征，放在了prompt中，所以在forward前，只需要将input_ids和attention_mask、labels放进GPU中，将Image留在CPU就行，节省一部分显存。  

```python
def main():
    args, config = parse_args_and_config()

    if args.save_example_data:
        save_example_dataset(args.save_example_data, example_image=args.example_image)
        print(f"已保存 MiniLLaVA DPO 示例数据: {args.save_example_data}")
        return

    accelerator = Accelerator()
    set_seed(int(config["MISC"]["SEED"]))
    accelerate_set_seed(int(config["MISC"]["SEED"]))

    policy_model = MiniLlavaModel(args.config)
    ref_model = MiniLlavaModel(args.config)
    load_checkpoint(policy_model, args.checkpoint)
    load_checkpoint(ref_model, args.checkpoint)
    freeze_model(ref_model)
    policy_model.train()

    train_config = config["DATA"]["TRAIN_DATASET"]
    dataset = LlavaDPODataset(
        data_file=args.train_file,
        image_root=args.image_root,
        max_samples=args.max_samples,
        example_image=args.example_image,
        toy_data=args.toy_data,
    )
    collator = LlavaDPOCollator(
        tokenizer=policy_model.language_decoder.tokenizer,
        max_length=int(train_config["MAX_LENGTH"]),
        max_turns=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(train_config["NUM_WORKERS"]),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = build_optimizer(policy_model, config)
    num_epochs = int(config["TRAINING"]["SCHEDULER"]["NUM_EPOCHS"])
    policy_model, ref_model, optimizer, dataloader = accelerator.prepare(
        policy_model,
        ref_model,
        optimizer,
        dataloader,
    )
    total_steps = len(dataloader) * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    log_steps = int(config["TRAINING"]["LOGGING"]["LOG_STEPS"])
    log_dir = config["TRAINING"]["LOGGING"].get("LOG_DIR", "week15_safety_alignment/outputs/logs")
    train_log_path = os.path.join(log_dir, "train_dpo.jsonl")
    save_steps = int(config["TRAINING"]["CHECKPOINT"]["SAVE_STEPS"])
    save_dir = os.path.join(config["TRAINING"]["CHECKPOINT"]["SAVE_DIR"], "dpo")
    max_norm = float(config["TRAINING"]["GRAD_CLIP"]["MAX_NORM"])

    if accelerator.is_main_process:
        print(f"启用 MiniLLaVA DPO 训练: n={len(dataset)}, beta={args.beta}")
        append_jsonl(train_log_path, {
            "event": "dpo_train_start",
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "train_file": args.train_file,
            "checkpoint": args.checkpoint,
            "num_epochs": num_epochs,
            "total_steps": total_steps,
            "beta": args.beta,
        })

    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            chosen = move_tensor_batch_to_device(batch["chosen"], accelerator.device)
            rejected = move_tensor_batch_to_device(batch["rejected"], accelerator.device)

            policy_chosen_logps = sequence_log_probs(policy_model, chosen)
            policy_rejected_logps = sequence_log_probs(policy_model, rejected)
            with torch.no_grad():
                ref_chosen_logps = sequence_log_probs(ref_model, chosen)
                ref_rejected_logps = sequence_log_probs(ref_model, rejected)

            loss, logits, rewards_chosen, rewards_rejected = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=args.beta,
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(policy_model.parameters(), max_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
            margin_value = accelerator.gather_for_metrics(logits).mean().item()
            chosen_reward = accelerator.gather_for_metrics(rewards_chosen).mean().item()
            rejected_reward = accelerator.gather_for_metrics(rewards_rejected).mean().item()

            if global_step % log_steps == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch + 1}/{num_epochs} step={global_step} "
                    f"dpo_loss={loss_value:.4f} margin={margin_value:.4f} lr={lr:.8f}"
                )
                append_jsonl(train_log_path, {
                    "event": "dpo_train_step",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "step": global_step,
                    "loss": loss_value,
                    "margin": margin_value,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "lr": lr,
                })

            if global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(accelerator, policy_model, optimizer, scheduler, global_step, save_dir)

    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, policy_model, optimizer, scheduler, global_step, save_dir)
    if accelerator.is_main_process:
        append_jsonl(train_log_path, {
            "event": "dpo_train_end",
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": global_step,
            "checkpoint_dir": save_dir,
        })
```
训练过程中，观察dpo的loss在降低，而margin在升高，说明模型在逐渐学会更偏好chosen。  

```text
epoch 1/3 step=10 dpo_loss=0.6914 margin=0.0355 lr=0.00000800
epoch 1/3 step=20 dpo_loss=0.6930 margin=0.0032 lr=0.00001600
epoch 1/3 step=30 dpo_loss=0.6931 margin=0.0009 lr=0.00002400
epoch 1/3 step=40 dpo_loss=0.6925 margin=0.0129 lr=0.00003200
epoch 1/3 step=50 dpo_loss=0.6915 margin=0.0321 lr=0.00004000
epoch 1/3 step=60 dpo_loss=0.6928 margin=0.0077 lr=0.00004800
epoch 1/3 step=70 dpo_loss=0.6920 margin=0.0235 lr=0.00005600
epoch 1/3 step=80 dpo_loss=0.7080 margin=-0.2949 lr=0.00006400
······
poch 1/3 step=3080 dpo_loss=1.2593 margin=-9.2550 lr=0.00007997
epoch 1/3 step=3090 dpo_loss=0.0143 margin=42.3839 lr=0.00007997
epoch 1/3 step=3100 dpo_loss=0.0716 margin=26.0059 lr=0.00007997
epoch 1/3 step=3110 dpo_loss=0.0966 margin=22.8842 lr=0.00007997
epoch 1/3 step=3120 dpo_loss=0.0117 margin=44.4317 lr=0.00007997
epoch 1/3 step=3130 dpo_loss=7.2198 margin=-72.1906 lr=0.00007997
epoch 1/3 step=3140 dpo_loss=5.9690 margin=-59.6642 lr=0.00007997
epoch 1/3 step=3150 dpo_loss=0.0164 margin=41.0307 lr=0.00007997
epoch 1/3 step=3160 dpo_loss=0.3580 margin=8.4293 lr=0.00007997
epoch 1/3 step=3170 dpo_loss=0.0097 margin=46.3441 lr=0.00007997
epoch 1/3 step=3180 dpo_loss=0.0006 margin=74.5925 lr=0.00007997
epoch 1/3 step=3190 dpo_loss=0.0570 margin=28.3674 lr=0.00007997
```

---

## 4. GRPO：面向推理任务的轻量 RL

GRPO，全称 Group Relative Policy Optimization。

它可以理解成 PPO 的轻量变体，常用于数学、代码、推理任务。

PPO 通常需要 Value Model 来估计 advantage：

```text
Advantage = 当前回答比预期好多少
```

但训练 Value Model 很贵，也不稳定。

GRPO 的思路是：

```text
不用 Value Model，而是对同一个 prompt 采样多个回答，用组内相对 reward 来估计 advantage
```

---

### GRPO 流程

对同一个 prompt，模型生成多个答案：

```text
Prompt: 计算 23 + 58

Answer 1: 81       reward = 1
Answer 2: 71       reward = 0
Answer 3: 23+58=81 reward = 1
Answer 4: 80       reward = 0
```

计算这一组的平均 reward：

```text
mean_reward
```

然后每个回答的 advantage 近似为：

$$
A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
$$
直觉：

```text
比组内平均好 → 提高概率
比组内平均差 → 降低概率
```

---

### GRPO 适合什么任务？

GRPO 特别适合可验证任务：

- 数学题
- 代码题
- 单元测试
- 格式检查
- JSON 输出
- 工具调用
- 多步推理
- 逻辑推理

因为这些任务可以比较容易设计规则 reward，假如reward都是通过规则制定出来的，那么就不需要训练reward model：

```text
答案正确：1
答案错误：0
格式错误：-0.2
```

---

## PPO / DPO / GRPO 对比

| 方法 | 核心思想                     | 是否需要 Reward Model | 是否需要 rollout | 适合场景         |
| ---- | ---------------------------- | --------------------: | ---------------: | ---------------- |
| SFT  | 模仿标准答案                 |                    否 |               否 | 指令微调         |
| DPO  | 直接学习 chosen > rejected   |                    否 |               否 | 偏好对齐         |
| PPO  | 用 Reward Model 强化高分回答 |                    是 |               是 | 经典 RLHF        |
| GRPO | 组内比较，强化更好回答       |                  可选 |               是 | 数学、代码、推理 |

**总结**

```text
DPO：
给 chosen/rejected，直接学会更偏好 chosen。

PPO：
自己生成回答，Reward Model 给分，高分回答以后更容易生成。

GRPO：
同一道题生成多个答案，组内比较，强化比平均更好的答案。
```

为了让模型更安全，有一些更直接的工程上的做法，比如在prompt之前和模型回答之后添加一些屏蔽词：

```python
BLACKLIST_WORDS = [
    "暴力",
    "自杀",
    "毒品",
    "色情",
    "仇恨",
    "炸弹",
    "枪支",
    "诈骗",
    "身份证",
    "银行卡",
]

SAFE_REFUSAL = "抱歉，这个请求包含不安全内容，我不能继续生成相关回答。"
SAFE_OUTPUT_FALLBACK = "抱歉，模型输出触发了安全过滤，我不能展示这段回答。"


def find_blacklist_word(text, blacklist_words=None):
    """检查文本是否命中黑名单词，返回第一个命中的词。"""
    text = str(text or "").lower()
    for word in blacklist_words or BLACKLIST_WORDS:
        if str(word).lower() in text:
            return word
    return None


def input_filter(question):
    """推理前过滤用户输入，命中黑名单时不再调用模型。"""
    hit_word = find_blacklist_word(question)
    if hit_word:
        return False, hit_word
    return True, None


def output_filter(answer):
    """推理后过滤模型输出，命中黑名单时返回安全兜底文案。"""
    hit_word = find_blacklist_word(answer)
    if hit_word:
        return SAFE_OUTPUT_FALLBACK, hit_word
    return answer, None

```