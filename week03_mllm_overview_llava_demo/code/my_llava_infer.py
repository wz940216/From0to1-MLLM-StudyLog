import torch
from torch import nn
from transformers import CLIPImageProcessor
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration
)
from PIL import Image

# =========================
# 配置
# =========================
model_path = "models/llava-1.5-7b-hf"  # 本地 LLaVA 模型路径
device = "cuda"  # 使用 GPU
dtype = torch.float16  # 使用半精度加速推理

# =========================
# 手写 LLaVA
# =========================
class MyLLaVA(nn.Module):
    # 整体流程：
    # image → CLIP → patch features → projector → LLaMA embedding space
    # text → tokenizer → embedding
    # ↓
    # 拼接（用图像 embedding 替换 <image> token）
    # ↓
    # LLaMA → logits → decode
    def __init__(self, hf_model):
        super().__init__()

        # 直接复用 HuggingFace 模型里的子模块（已经包含训练好的权重）
        self.vision_tower = hf_model.vision_tower  # 图像编码器（CLIP）
        self.projector = hf_model.multi_modal_projector  # 多模态投影层
        self.llm = hf_model.language_model  # 语言模型（LlamaForCausalLM）
        self.config = hf_model.config  # 配置

    def encode_image(self, image_tensor):
        """
        输入:
            image_tensor: (B, 3, 336, 336)
        输出:
            (B, 576, 4096) 图像 patch 对应的 LLM embedding
        """
        outputs = self.vision_tower(
            image_tensor,
            output_hidden_states=True  # 输出所有中间层
        )

        # 取指定层的特征（LLaVA 配置里指定）
        feats = outputs.hidden_states[self.config.vision_feature_layer]

        # 去掉 CLS token，只保留 patch tokens
        feats = feats[:, 1:]

        # 通过 projector 映射到 LLM 的 embedding 维度
        feats = self.projector(feats)

        return feats

    def merge_embeddings(self, input_ids, image_embeds):
        """
        将 <image> token 替换为图像 embedding
        """
        # 获取文本 embedding
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        B, T, D = inputs_embeds.shape  # batch, seq_len, hidden_dim
        _, N, _ = image_embeds.shape   # image patch 数量

        image_token_id = self.config.image_token_index  # <image> token 对应的 id

        merged = []

        # 遍历 batch
        for b in range(B):
            cur = []
            # 遍历每个 token
            for t in range(T):
                if input_ids[b, t] == image_token_id:
                    # 如果是 <image>，插入整块图像 embedding（576个patch）
                    cur.append(image_embeds[b])
                else:
                    # 普通 token，保留原 embedding
                    cur.append(inputs_embeds[b, t:t+1])
            # 拼接当前序列
            cur = torch.cat(cur, dim=0)
            merged.append(cur)

        # 对 batch 内不同长度做 padding
        max_len = max(x.shape[0] for x in merged)

        padded = []
        for x in merged:
            pad_len = max_len - x.shape[0]
            if pad_len > 0:
                # 用 0 填充
                pad = torch.zeros(pad_len, D, device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            padded.append(x)

        # stack 成 batch
        return torch.stack(padded, dim=0)

    def forward(self, input_ids, image_tensor):
        # 图像编码
        image_embeds = self.encode_image(image_tensor)

        # 融合文本和图像 embedding
        inputs_embeds = self.merge_embeddings(input_ids, image_embeds)

        # 输入 LLM（直接用 embedding，不用 input_ids）
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            use_cache=False  # 不使用 KV cache（简单实现）
        )

        # 获取 logits（已经包含 lm_head）
        logits = outputs.logits 
        
        return logits


# =========================
# 生成函数（greedy 解码）
# =========================
@torch.no_grad()
def generate(model, tokenizer, input_ids, image_tensor, max_new_tokens=500):
    generated = input_ids  # 初始化生成序列

    for step in range(max_new_tokens):
        # 前向计算 logits
        logits = model(generated, image_tensor)

        # 取最后一个 token 的 logits，并选最大概率
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # 拼接到生成序列
        generated = torch.cat([generated, next_token], dim=-1)

        # debug: 打印解码过程
        # print(tokenizer.decode(generated[0], skip_special_tokens=True), end="\r")
        
        # 如果生成 EOS token，提前结束
        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated


# =========================
# 主函数
# =========================
def main():
    print("Loading HF model (for weights)...")
    # 加载 HuggingFace 官方模型（仅用于获取权重）
    hf_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)

    hf_model.eval()

    print("Building custom model...")
    # 构建我们自己手写的 LLaVA
    model = MyLLaVA(hf_model).to(device).eval()

    print("Loading tokenizer & processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 文本 tokenizer
    processor = CLIPImageProcessor.from_pretrained(model_path)  # 图像预处理器

    # =========================
    # 构造输入
    # =========================
    prompt = "USER: <image>\n请用中文描述这张图片。\nASSISTANT:"

    # tokenizer 编码
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # =========================
    # 加载图片
    # =========================
    image = Image.open("week03_mllm_overview_llava_demo/code/image.png").convert("RGB")
    
    # debug（可用于检查 processor 是否正常）
    # out = processor(images=image, return_tensors="pt")
    # print(type(out), out)
    
    # 图像转 tensor
    image_tensor = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(device, dtype)
    
    # print(image_tensor.shape)  # 应为 (1, 3, 336, 336)

    # =========================
    # 推理
    # =========================
    print("Generating...")
    output_ids = generate(model, tokenizer, input_ids, image_tensor)

    # 解码生成文本
    result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n===== RESULT =====")
    print(result)


if __name__ == "__main__":
    main()