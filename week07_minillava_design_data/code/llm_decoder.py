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
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device

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
    
    
    
    
    
    
    