import gradio as gr
import random
import os
import torch
import sys
from PIL import Image
    
# Add workspace root to sys.path so sibling packages can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

# 调用clip检索模块
from week04_clip_retrieval.code.clip_simple_image_retrieval import build_image_index, retrieve_topk

# -------------------
# Mock: 图片 → caption
# -------------------
def image_to_caption(img):
    return "A photo of something interesting."  # 未来替换模型
def batch_caption(images):
    results = []
    for i, img in enumerate(images):
        results.append(f"Caption for image {i+1}")
    return results
# -------------------
# Mock: 文本 → 图片检索
# -------------------

# 图库目录 加载图像特征
image_dir = os.path.join(ROOT_DIR, "dataset", "coco128", "images", "train2017")
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
image_index_path = os.path.join(ROOT_DIR, "week06_clip_blip_web_demo", "image_index.pt")
# 如果索引文件不存在，则构建索引并保存
if not os.path.exists(image_index_path):
    print("Building image index...")
    image_features = build_image_index(image_paths)
    torch.save(image_features, image_index_path)
    print("Image index built with shape:", image_features.shape)
else:
    print("Loading image index from file...")
    image_features = torch.load(image_index_path)
    print("Image index loaded with shape:", image_features.shape)


def text_to_images(text, k):
    # 使用CLIP模型进行图像检索
    topk_indices = retrieve_topk(text, image_features, image_paths, k)
    return [Image.open(topk["image_path"]).convert("RGB") for topk in topk_indices]

# -------------------
# UI
# -------------------
with gr.Blocks() as demo:
    gr.Markdown("# Image ↔ Text Demo")
    with gr.Tab("Image → Caption"):
        gallery_input = gr.Gallery(
        label="Upload Images",
        type="pil"
        )
        btn = gr.Button("Generate Captions")
        text_output = gr.JSON(label="Captions")
        btn.click(
            fn=batch_caption,
            inputs=gallery_input,
            outputs=text_output
        )

    with gr.Tab("Text → Image"):
        text_input = gr.Textbox(label="Enter description")
        k = gr.Slider(1, 10, value=3, step=1, label="Number of images")
        btn2 = gr.Button("Search Images")
        # gallery = gr.Gallery()
        gallery = gr.Gallery(columns=3, rows=2, object_fit="scale-down", scale=1)
        btn2.click(fn=text_to_images,
                   inputs=[text_input, k],
                   outputs=gallery, show_progress=True)

demo.launch()