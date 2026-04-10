from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
# ==============================
# 1. 加载模型和预处理器
# ==============================

# BlipProcessor：负责把图片转换成模型可以理解的输入格式（如 tensor）
processor = BlipProcessor.from_pretrained("models/blip-image-captioning-base")

# BlipForConditionalGeneration：BLIP 图像描述生成模型
model = BlipForConditionalGeneration.from_pretrained("models/blip-image-captioning-base")


# ==============================
# 2. 加载图片
# ==============================

# 这里使用本地图片路径（也可以是网络图片）
img_url = "week05_blip_blip2_caption/code/000000039769.jpg"

# 使用 PIL 打开图片，并转换为 RGB 格式（模型要求）
image = Image.open(img_url).convert("RGB")


# ==============================
# 3. 预处理输入
# ==============================

# 将图片转换为 PyTorch tensor，并自动做 resize / normalize 等处理
# return_tensors="pt" 表示返回 PyTorch 格式
inputs = processor(images=image, return_tensors="pt")

# inputs = processor(images=image, text="用中文描述这张图片：", return_tensors="pt")
# ==============================
# 4. 将模型和数据移动到 GPU
# ==============================

# 把模型加载到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 把输入数据也移动到 GPU
inputs = {k: v.to(device) for k, v in inputs.items()}


# ==============================
# 5. 生成 caption
# ==============================

# generate 是文本生成核心函数
out = model.generate(
    **inputs,          # 输入图片特征
    max_length=50,     # 生成文本的最大长度
    num_beams=5,       # Beam Search
    temperature=1.0    # 控制随机性（1.0为默认，越小越保守）
)


# ==============================
# 6. 解码输出
# ==============================

# 将模型输出的 token id 转换为自然语言文本
caption = processor.decode(out[0], skip_special_tokens=True)


# ==============================
# 7. 打印结果
# ==============================

print("Caption:", caption)