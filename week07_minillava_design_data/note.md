# minillava_design_data

本篇是minillava系列的起始篇，week07开始将从零搭建一个mini的llava模型，本周我们先设计minillava的模型框架，下载必要的微调数据集，串好数据流，验证forward功能。并将参数配置到config.yaml中。

## llava结构回顾
LLaVA是一系列结构极简的多模态大模型。不同于Flamingo的交叉注意力机制、BLIP系列的Q-Former，LLaVA直接使用简单的线性层将视觉特征映射为文本特征，在一系列的多模态任务上取得了很好的效果。

LLaVA 的核心结构是“CLIP视觉编码器 + 线性映射层 + 大语言模型（LLM）”，通过简单高效的方式实现视觉与语言特征对齐。

**视觉编码器**（Vision Encoder）
使用预训练的 CLIP 模型提取图像特征。 

**线性映射层**（Projection Layer）
将视觉特征映射到与 LLM 词嵌入空间相同的维度
该层是轻量级的MLP或全连接层，用于实现视觉与文本特征的对齐。
映射后的视觉 token embedding 与文本 token embedding 合并，作为 LLM 的输入。 

**大语言模型**（LLM）
通常使用Vicuna或LLaMA系列作为语言解码器。 
接收融合后的视觉-文本序列，生成多模态指令响应或文本描述。
支持多轮对话和任务指令跟随，训练目标为最大似然概率预测每个token。 

## 自己搭建一个minillava

### Vision Encoder
vision encoder我们采用week04中学习的clip模型，选用models/clip-vit-base-patch16做为minillava的图像特征提取。  

在搭建时，有两点需要注意：  

1、取clip中vision encoder的last_hidden_state输出，并且去掉cls的整体patch特征输出，只保留图像patch的特征输出，outputs中还有一个pooler_output是将cls token的输出进行layernorm之后的结果，代表整张图的全局特征，不适合用在理解图像，反而patch中的特征更符合llm理解。  

2、给模型增加freeze的功能，确保训练时可以通过配置文件冻结或解冻视觉编码器的模型参数。  

代码如下：

```python
from PIL import Image
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

class VisionEncoder(torch.nn.Module):
    def __init__(self, model_path, freeze=True, device="cuda"):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_path).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_path)
        self.device = device
        self.freeze = freeze
        # 冻结vision encoder参数，只训练projector和llm decoder的参数
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        if self.freeze:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
        else:
            outputs = self.vision_model(pixel_values=pixel_values)

        # (B, 197, 768)
        # 这里取clip中vision encoder的last_hidden_state输出，并且去掉cls的整体patch特征输出，只保留图像patch的特征输出
        # outputs中还有一个pooler_output是将cls token的输出进行layernorm之后的结果，代表整张图的全局特征，不适合用在理解图像
        # 反而patch中的特征更符合llm理解。
        last_hidden_state = outputs.last_hidden_state

        # 去掉 CLS → (B, 196, 768)
        patch_features = last_hidden_state[:, 1:, :]

        return patch_features
    
if __name__ == "__main__":
    
    vision_encoder = VisionEncoder(model_path="models/clip-vit-base-patch16", device="cuda")
    images_file = "dataset/coco128/images/train2017/000000000009.jpg"
    image = Image.open(images_file).convert("RGB")
    features = vision_encoder([image])
    print(features.shape)

```

### projector
projector一般为两层全连接层，并增加LayerNorm把不同模态的特征标准化到一个可对齐、可训练、稳定的空间里。  
视觉特征通过projector将特征映射到能和文本embedding拼接的尺寸。

代码如下：

```python
class Projector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x
```

### llm decoder
llm decoder中，我们选用week02中实验过的models/Qwen1.5-1.8B模型。

这里需要注意：  

1、类中要有get_input_embeddings函数来返回输入token的embedding信息。  
因为我们需要将图像信息和文本信息拼接到一起输入到llm中，简单的tokenid和图像信息无法拼接到一起。
所以可以调用模型的  

```python
self.model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,labels=labels)  
```
直接将图像和文本输入的embedding拼接好后传给llm模型。  
在拼接时会用到get_input_embeddings来返回文本的embedding信息，用于和图像embedding拼接。  

2、构建labels时，应将原始label的长度扩展到image token和text token长度的总和，并将image token处的labels设置为-100。文本部分用实际 token id 。  

因果语言模型的训练目标是预测文本部分的下一个token，所以图片部分的 labels 设置为 -100，表示这些位置的损失将被忽略。这样模型在训练时只会关注文本部分的预测，而不会受到图片部分的影响。  

-100的来源：在 HuggingFace / PyTorch 训练中： CrossEntropyLoss(ignore_index=-100)。  

即使vision encoder的参数更新，图片的label依然是-100，因为图片patch的token id本身就是没有意义的，不应该对模型预测文本的能力产生影响。  

3、构建attention_mask时，直接构建一个长度为image token和text token总和的全1向量即可。代表图片和文本的信息都是有效输入。  

这里的attention_mask区别于label中的使用-100的强制忽略，attention_mask决定llm在做self attention时是否使用该位置的tensor，所以图片和文本信息的mask全为1，代表都要做self attention。  

而label中的-100是在计算交叉熵时，强制忽略对应tensor对loss的影响，最终目的是让图像的token不用来预测下一个token是什么。

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class Projector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class LLMDecoder(torch.nn.Module):
    def __init__(self, model_path, freeze=False, device="cuda"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.freeze = freeze
        self.device = device
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, inputs_embeds, attention_mask, labels=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    
if __name__ == "__main__":
    
    llm_decoder = LLMDecoder(model_path="models/Qwen1.5-1.8B")
    projector = Projector(input_dim=768, hidden_dim=2048, output_dim=llm_decoder.model.config.hidden_size).to(llm_decoder.device)
    # 模拟图片特征输入
    dummy_input = torch.randn(1, 196, 768).to(llm_decoder.device)  # Simulating projected vision features
    # 图片特征经过project层，对齐到llm输入维度
    projected_input = projector(dummy_input)  # Project to LLM input dimension
    print(projected_input.shape)  # Should be (1, 196, model.config.hidden_size)
    
    # 输入文本进行tokenizer，获取input_ids，并通过get_input_embeddings获取文本的embedding
    tokenizer = llm_decoder.tokenizer
    text = "What is in the image?"
    text_inputs = tokenizer(text, return_tensors="pt").to(llm_decoder.device)
    text_input_ids = text_inputs.input_ids
    text_attention_mask = text_inputs.attention_mask
    text_embeddings = llm_decoder.get_input_embeddings()(text_input_ids)
    print(text_embeddings.shape)  # Should be (1, seq_len, model.config.hidden_size)
    
    # Combine projected vision features and text embeddings
    # 直接在特征维度上对图片和文本的embedding进行拼接，并将mask延长，适应新的输入长度。
    # mask中图片部分也全是1，表示图片和文本都是有效输入。
    combined_inputs = torch.cat([projected_input, text_embeddings], dim=1)
    combined_attention_mask = torch.cat([torch.ones(projected_input.size(0), projected_input.size(1)).to(llm_decoder.device), text_attention_mask], dim=1)  
    print(combined_inputs.shape)  # Should be (1, 196 + seq_len, model.config.hidden_size)
    print(combined_attention_mask.shape)  # Should be (1, 196 + seq_len)
    
```

### minillava
将以上三个部分串起来可实现一个最简化版本的minillava模型。  

```python
import torch
from vision_encoder import VisionEncoder
from llm_decoder import LLMDecoder, Projector
import yaml

class MiniLlavaModel(torch.nn.Module):
    def __init__(self, config_path):
        super(MiniLlavaModel, self).__init__()
        self.config = self.load_config(config_path)
        self.vision_encoder = VisionEncoder(model_path=self.config['MINILLAVA']['VISION_ENCODER']['MODEL_PATH'], freeze=self.config['MINILLAVA']['VISION_ENCODER']['FREEZE'], device=self.config['DEVICE'])
        self.language_decoder = LLMDecoder(model_path=self.config['MINILLAVA']['LLM_DECODER']['MODEL_PATH'], device=self.config['DEVICE'])
        self.projector = Projector(input_dim=self.config['MINILLAVA']['PROJECTOR']['INPUT_DIM'], hidden_dim=self.config['MINILLAVA']['PROJECTOR']['HIDDEN_DIM'], output_dim=self.language_decoder.model.config.hidden_size).to(self.config['DEVICE'])
        self.device = self.config['device']

    def load_config(self,config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def forward(self, images, texts):
        # 图片特征
        image_features = self.vision_encoder(images)
        projected_image_features = self.projector(image_features)
        
        # 文本特征
        text_inputs = self.language_decoder.tokenizer(texts, return_tensors="pt").to(self.device)
        text_embeddings = self.language_decoder.get_input_embeddings()(text_inputs.input_ids)
        
        # attention mask
        image_attention_mask = torch.ones(projected_image_features.size(0),projected_image_features.size(1)).to(self.device)
        text_attention_mask = text_inputs.attention_mask
        combined_attention_mask = torch.cat([image_attention_mask, text_attention_mask], dim=1)
        
        # 拼接多模态embeding
        combined_inputs = torch.cat([projected_image_features, text_embeddings], dim=1)
        
        # 创建 labels: 图片部分用 -100（忽略）, 文本部分用实际 token id 
        # 因果语言模型的训练目标是预测文本部分的下一个 token，所以图片部分的 labels 设置为 -100，表示这些位置的损失将被忽略。
        # 这样模型在训练时只会关注文本部分的预测，而不会受到图片部分的影响。
        # 确保labels的长度和 combined_inputs 的序列长度一致。
        # -100的来源：在 HuggingFace / PyTorch 训练中： CrossEntropyLoss(ignore_index=-100)
        # 即使vision encoder的参数更新，图片的label依然是-100，因为图片patch的token id本身就是没有意义的，不应该对模型预测文本的能力产生影响。
        image_labels = torch.full((projected_image_features.size(0), projected_image_features.size(1)), -100, dtype=torch.long).to(self.device)
        combined_labels = torch.cat([image_labels, text_inputs.input_ids], dim=1)
        
        outputs = self.language_decoder(inputs_embeds=combined_inputs, attention_mask=combined_attention_mask, labels=combined_labels)
        
        return outputs
    
if __name__ == "__main__":
    model = MiniLlavaModel(config_path="week07_minillava_design_data/code/config.yaml")
    from PIL import Image
    image = Image.open("dataset/coco128/images/train2017/000000000009.jpg").convert("RGB")
    dummy_images = [image,image,image]  # Simulating a batch of 3 images
    dummy_texts = ["What is in the image?"] * 3  # Simulating a batch of 3 identical questions
    outputs = model(dummy_images, dummy_texts)
    
    print(outputs.logits.shape) #logits.shape = (batch_size, seq_len, vocab_size) vocab_size 是词表大小
    # torch.Size([3, 202, 151936])
    print(outputs.past_key_values[0][0].shape) # past_key_values 是 Transformer attention 的 KV 缓存（Key-Value cache）
    # past_key_values = [
    #     (k1, v1),   # layer 0
    #     (k2, v2),   # layer 1
    #     ...
    #     ]
    # 每个 k/v shape：
    # (batch, num_heads, seq_len, head_dim)
    # torch.Size([3, 16, 202, 128])
    
    print(outputs.loss)
    # tensor(9.4443, device='cuda:0', grad_fn=<NllLossBackward0>)
    
```

比较有意思的是，我在调试代码过程中，发现llm模型的outputs中除了logits和loss，还有一个past_key_values。代表self attention时的k v缓存，每一次前向，llm只做新增token的attention，而将之前计算好的k和v存在past_key_values中，可以加快模型的推理速度。后续的week17会进一步一起探讨和研究llm的部署加速策略。  

总体来说llava的结构非常简洁，但拼接过程中有些细节需要重点处理，例如提取哪些视觉特征，如何拼接视觉和文本embedding，如何制作label和attentionmask等。