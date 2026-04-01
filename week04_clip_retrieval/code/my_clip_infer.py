import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. 加载模型
model = CLIPModel.from_pretrained("models/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch16")

# 2. 准备数据
url = "week04_clip_retrieval/code/000000039769.jpg"
image = Image.open(url)

labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# 3. 处理输入
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# 4. 手动前向推理
with torch.no_grad():
    # 图像特征[1,512]
    image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            
    # 文本特征[3,512]
    text_features = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

# 5. 归一化（CLIP 的关键步骤）
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 6. 手动计算相似度（cosine similarity）[1,3]
logits = image_features @ text_features.T   # (1, num_labels)

# CLIP 默认有一个 temperature scaling
logit_scale = model.logit_scale.exp()
logits = logits * logit_scale

# 7. softmax 得到概率
probs = logits.softmax(dim=1)

# 8. 选最大
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = labels[most_likely_idx]

print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")