import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import time

# ======================
# 1. 加载模型
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("models/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch16")

model.eval()

# ======================
# 2. 构建图库（预编码）
# ======================
def build_image_index(image_paths):
    image_features_list = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            feat = model.get_image_features(**inputs)

        # 归一化
        feat = feat / feat.norm(dim=-1, keepdim=True)

        image_features_list.append(feat)

    # 拼接成矩阵 [N, 512]
    image_features = torch.cat(image_features_list, dim=0)

    return image_features


# ======================
# 3. 文本查询
# ======================
def encode_text(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # [1, 512]


# ======================
# 4. 检索 Top-K
# ======================
def retrieve_topk(query, image_features, image_paths, topk=3):
    text_feat = encode_text(query)

    # cosine similarity
    logits = text_feat @ image_features.T  # [1, N]

    # 温度系数
    logit_scale = model.logit_scale.exp()
    logits = logits * logit_scale

    # Top-K
    topk_scores, topk_indices = torch.topk(logits, k=topk, dim=1)

    results = []
    for score, idx in zip(topk_scores[0], topk_indices[0]):
        results.append({
            "image_path": image_paths[idx],
            "score": score.item()
        })

    return results


# ======================
# 5. 示例运行
# ======================
if __name__ == "__main__":
    # 假设你的图库
    image_dir = "dataset/coco128/images/train2017"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

    print("Building image index...")
    image_features = build_image_index(image_paths)
    # torch.save(image_features, "image_index.pt")
    print("Image index built with shape:", image_features.shape)
    # 查询
    query = "a photo of a cat"
    
    for i in range(3):
        t1 = time.time()
        results = retrieve_topk(query, image_features, image_paths, topk=3)
        print(f"Retrieval done in {time.time() - t1:.4f} seconds")

    print("\nQuery:", query)
    print("Top-K results:")
    for r in results:
        print(f"{r['image_path']}  score={r['score']:.4f}")