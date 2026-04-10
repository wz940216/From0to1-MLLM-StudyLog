from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from PIL import Image
import torch

# 基础模型路径（BLIP Image Captioning Base）
base_model_name = "models/blip-image-captioning-base"

# LoRA 训练检查点目录（已训练好的 LoRA 权重）
lora_model_dir  = "week05_blip_blip2_caption/outputs/checkpoint-3753"   # 训练好的 LoRA 保存目录

# 加载 tokenizer
processor = BlipProcessor.from_pretrained(base_model_name)
tokenizer = processor.tokenizer

# 加载基础模型
base_model = BlipForConditionalGeneration.from_pretrained(base_model_name)

# 加载 LoRA 权重到基础模型上
model = PeftModel.from_pretrained(base_model, lora_model_dir)

# 把模型加载到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device)

# 这里使用本地图片路径（也可以是网络图片）
img_url = "week05_blip_blip2_caption/code/000000039769.jpg"

# 使用 PIL 打开图片，并转换为 RGB 格式（模型要求）
image = Image.open(img_url).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
# inputs = processor(images=image, text="用中文描述这张图片：", return_tensors="pt") 中文直接拉胯了

# 把输入数据也移动到 GPU
inputs = {k: v.to(device) for k, v in inputs.items()}

# generate 是文本生成核心函数
out = model.generate(
    **inputs,          # 输入图片特征
    max_length=50,     # 生成文本的最大长度
    num_beams=5,       # Beam Search
    temperature=1.0    # 控制随机性（1.0为默认，越小越保守）
)

# 解码输出
# 将模型输出的 token id 转换为自然语言文本
caption = processor.decode(out[0], skip_special_tokens=True)
print("Caption:", caption)