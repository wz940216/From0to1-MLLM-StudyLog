import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

class Projector(nn.Module):
    """把 CLIP 视觉特征映射到 LLM 词向量空间的两层 MLP。

    CLIP 的隐藏维度通常是 768，而 Qwen1.5-1.8B 的 hidden_size 是 2048。
    LLM 只能接收自己 hidden_size 维度的 inputs_embeds，因此必须加一个 projector
    完成维度对齐，这也是 MiniLLaVA 中最核心的跨模态连接层。
    """

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
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x


class LLMDecoder(torch.nn.Module):
    def __init__(self, model_path, r=8, lora_alpha=32, lora_dropout=0.1, freeze=False, device="cuda"):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        
        # trust_remote_code=True 兼容部分国产模型仓库的自定义实现；本地官方 Qwen 也可正常加载。
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(self.device)
        
        if r != 'None':
            # LoRA 参数配置
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            # 包装成 PEFT LoRA 模型
            self.model = get_peft_model(self.model, self.peft_config)
            freeze = False  # 使用 LoRA 时通常不冻结原模型参数，允许它们在训练中更新以配合 LoRA 调整。
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            # 因果语言模型常常没有 pad token。训练 batch padding 时复用 eos 最稳妥。
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.freeze = freeze
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def forward(self, inputs_embeds, attention_mask, labels=None, use_cache=False):
        """直接接收拼好的多模态 embedding。

        MiniLLaVA 的图片部分没有 token id，因此不能走普通 input_ids 路径；
        需要由外部先把视觉 embedding 和文本 embedding 拼好，再传给 LLM。
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache
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
    
    
    
    
    
    
    
