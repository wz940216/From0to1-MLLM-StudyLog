import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ======================
# 1. 加载模型
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("models/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch16")

model.eval()

# ======================
# 2. 构造文本 prompt
# ======================
def build_prompts(class_names, template="a photo of a {}"):
    return [template.format(name) for name in class_names]


# ======================
# 3. 编码文本（一次性）
# ======================
def encode_text(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # 归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # [num_classes, 512]


# ======================
# 4. 编码图像
# ======================
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features  # [1, 512]


# ======================
# 5. 零样本分类
# ======================
def zero_shot_classify(image_path, class_names, topk=3):
    # 构造 prompt
    prompts = build_prompts(class_names)

    # 编码
    text_features = encode_text(prompts)
    image_features = encode_image(image_path)

    # 相似度
    logits = image_features @ text_features.T  # [1, num_classes]

    # CLIP temperature
    logit_scale = model.logit_scale.exp()
    logits = logits * logit_scale

    # 概率
    probs = logits.softmax(dim=-1)

    # Top-K
    topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)

    results = []
    for prob, idx in zip(topk_probs[0], topk_indices[0]):
        results.append({
            "label": class_names[idx],
            "prob": prob.item()
        })

    return results


# ======================
# 6. 示例
# ======================
if __name__ == "__main__":
    image_path = "dataset/coco128/images/train2017/000000000074.jpg"

    class_names = [
        "cat",
        "dog",
        "car",
        "truck",
        "airplane",
        "person"
    ]

    results = zero_shot_classify(image_path, class_names, topk=3)

    print("Top predictions:")
    for r in results:
        print(f"{r['label']}: {r['prob']:.4f}")