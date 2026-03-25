import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# 1. 模型和处理器
model_id = "models/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# 2. 准备图片和文本
image = Image.open("week03_mllm_overview_llava_demo/code/image.png").convert("RGB") 
prompt = "USER: <image>\n请用中文描述这张图片。\nASSISTANT:"

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(device)

# 3. 推理
generate_ids = model.generate(
    **inputs,
    max_new_tokens=200
)

# 4. 解码输出
output = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(output)
