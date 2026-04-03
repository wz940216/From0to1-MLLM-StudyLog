# BLIP / BLIP-2 Caption
## BLIP
BLIP（Bootstrapped Language-Image Pretraining），源自《BLIP: Bootstrapping Language-Image Pre-training for Unifified Vision-Language Understanding and Generation》

![blip](blip.png)
### MED
全称为Multimodal Mixture of Encoder-Decoder，是一个可以完成三个任务的复合型模型。

MED主要由四部分组成。图中相同颜色的部分共用参数。 
与CLIP类似，VIT作图像编码。
图像编码器的输出会流向以下三个不同的结构。

#### 1、ITC
这个结构类似BERT使用的编码器。将文本开头加入一个[CLS]token之后输入编码器进行编码，然后其输出与图片编码器的输出进行image-text contrastive任务。

这个任务本质就是对比学习的做法，ITC loss借鉴了ALBEF中的做法，引入动量编码器。
其实作者准备了两个编码器

1️⃣ 主编码器（student）正常训练，参与反向传播，参数：θ

2️⃣ 动量编码器（teacher）不参与反向传播，参数：θ_m，更新方式：

$$
\theta_m \leftarrow m \cdot \theta_m + (1 - m) \cdot \theta
$$
m 通常是 0.995 / 0.999

使用软标签可以让模型学习到更细粒度的语义信息，而不是简单地二值化判断

#### 2、ITM
这个编码器与上一个的唯一不同点就在于多加了一个cross attention(CA)操作。将图片编码器的输出embedding作为query，文本编码器中self attention(SA)之后的embedding作为key和value进行CA操作，这样可以学习到图片和文本的多模态表达以用于捕捉更加精细的视觉与语言之间的对应关系。

这个结构对应的则是image-text matching任务。这是一个二分类任务，通过添加一个线性层输出positive或者negative来表示图片和文本是否匹配。输入时，须在开头处添加[Encode]token表示起始

#### 3、LM
双向自注意力机制改为了因果自注意力机制，即Transformer中的decoder只与之前出现的token进行attention操作。这个结构是用于基于给定图片生成对应的文本描述。

它对应语言模型Language Modeling任务，损失函数是交叉熵。同样地，输入文本开头需要添加一个[Decode]token表示开头，结尾需要添加一个[EOS]字符。

训练方式和标准的文本生成模型（类似 GPT）是一样的，只不过它是以图像特征 + 文本前缀作为条件来生成文本。label就是文本序列本身（右移后的 token），也就是做teacher forcing的next-token prediction。

#### CapFilt
作者利用标注精准的数据集训练了一个BLIP后，利用BLIP的图像描述能力，在网图（标注不太准的数据集）中生成了较为准确的图像描述句子。再利用BLIP的图像文本比对能力做一次筛选。用洗过的数据又训了一次BLIP。也就是作者提到的boostrapping，自举法，先用准数据训练模型，用该模型洗数据后，再用于模型训练。

## BLIP-2

