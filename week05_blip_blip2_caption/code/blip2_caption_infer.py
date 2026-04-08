from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

# ==============================
# 1. 加载模型和预处理器
# ==============================

processor = Blip2Processor.from_pretrained("models/blip2-flan-t5-xl")

model = Blip2ForConditionalGeneration.from_pretrained(
    "models/blip2-flan-t5-xl",
    torch_dtype=torch.float16  # 半精度
)

# ==============================
# 2. 加载图片
# ==============================

img_url = "week05_blip_blip2_caption/code/000000039769.jpg"
image = Image.open(img_url).convert("RGB")

# ==============================
# 3. 预处理输入
# ==============================

# BLIP-2 支持 prompt 可以引导生成
# prompt = "Question: What is in the image? Answer:"

prompt = "Question: Describe the image in detail. Answer:"

inputs = processor(images=image, text=prompt, return_tensors="pt")

# ==============================
# 4. 将模型和数据移动到 GPU
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

inputs = {k: v.to(device) for k, v in inputs.items()}

# ==============================
# 5. 生成 caption
# ==============================

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=50,   # 注意这里是 max_new_tokens BLIP-2常用
        num_beams=5,
        temperature=1.0
    )

# ==============================
# 6. 解码输出
# ==============================

caption = processor.decode(out[0], skip_special_tokens=True)

# ==============================
# 7. 打印结果
# ==============================

print("Caption:", caption)