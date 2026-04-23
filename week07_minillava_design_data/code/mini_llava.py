import torch
from vision_encoder import VisionEncoder
from llm_decoder import LLMDecoder, Projector

class MiniLlavaModel(torch.nn.Module):
    def __init__(self, vision_encoder_model_path, language_decoder_model_path, device="cuda"):
        super(MiniLlavaModel, self).__init__()
        self.vision_encoder = VisionEncoder(model_path=vision_encoder_model_path, device=device)
        self.language_decoder = LLMDecoder(model_path=language_decoder_model_path, device=device)
        self.projector = Projector(input_dim=768, hidden_dim=2048, output_dim=self.language_decoder.model.config.hidden_size).to(device)
        self.device = device
        
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
    model = MiniLlavaModel(vision_encoder_model_path="models/clip-vit-base-patch16", language_decoder_model_path="models/Qwen1.5-1.8B", device="cuda")
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
    
    