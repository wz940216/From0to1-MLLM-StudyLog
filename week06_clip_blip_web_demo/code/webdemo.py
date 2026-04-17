import gradio as gr
import random
import os
import torch
import sys
import io
import base64
from PIL import Image
    
# Add workspace root to sys.path so sibling packages can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

# 调用clip检索模块
from week04_clip_retrieval.code.clip_simple_image_retrieval import build_image_index, retrieve_topk
# 调用blip caption模块
from week05_blip_blip2_caption.code.blip_caption_infer import BLIPCaptioner

# -------------------
# Mock: 图片 → caption
# -------------------
# 初始化BLIPCaptioner实例，提前加载模型
captioner = BLIPCaptioner(model_path="models/blip-image-captioning-base")

def pil_to_data_uri(image):
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def normalize_gallery_image(item):
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return item[0]
    return item


def image_to_caption(img):
    image = normalize_gallery_image(img)
    return captioner.generate_caption(image)


def batch_caption(images):
    cards = []
    for idx, img_item in enumerate(images, start=1):
        image = normalize_gallery_image(img_item)
        caption_text = captioner.generate_caption(image)
        image_uri = pil_to_data_uri(image)
        cards.append(
            f"""
            <div class=\"caption-card\">
                <img src=\"{image_uri}\" />
                <div class=\"caption-overlay\">{caption_text}</div>
            </div>
            """
        )

    return f"""
    <style>
        .caption-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
        }}
        .caption-card {{
            position: relative;
            overflow: hidden;
            border-radius: 12px;
            background: #111;
        }}
        .caption-card img {{
            width: 100%;
            height: auto;
            display: block;
            object-fit: cover;
        }}
        .caption-overlay {{
            position: absolute;
            top: 10px;
            left: 10px;
            right: 10px;
            padding: 8px 12px;
            background: rgba(0, 0, 0, 0.65);
            color: #fff;
            font-size: 0.95rem;
            border-radius: 8px;
            line-height: 1.4;
            max-height: 5.5em;
            overflow: hidden;
        }}
    </style>
    <div class=\"caption-grid\">
        {''.join(cards)}
    </div>
    """
# -------------------
# Mock: 文本 → 图片检索
# -------------------

# 图库目录 加载图像特征
image_dir = os.path.join(ROOT_DIR, "dataset", "COCOCaption", "val2017")
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


def image_to_data_uri(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
    return f"data:{mime};base64,{data}"


def text_to_images(text, k):
    # 使用CLIP模型进行图像检索
    topk_indices = retrieve_topk(text, image_features, image_paths, k)

    cards = []
    for idx, item in enumerate(topk_indices, start=1):
        image_uri = image_to_data_uri(item["image_path"])
        caption_text = f"Rank {idx}_{item['score']}: {os.path.basename(item['image_path'])}"
        cards.append(
            f"""
            <div class=\"search-card\">
                <img src=\"{image_uri}\" />
                <div class=\"search-caption\">{caption_text}</div>
            </div>
            """
        )

    return f"""
    <style>
        .search-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }}
        .search-card {{
            position: relative;
            overflow: hidden;
            border-radius: 10px;
            background: #111;
        }}
        .search-card img {{
            width: 100%;
            height: auto;
            display: block;
            object-fit: cover;
        }}
        .search-caption {{
            position: absolute;
            top: 8px;
            left: 8px;
            padding: 6px 10px;
            background: rgba(0, 0, 0, 0.6);
            color: white;
            font-size: 0.9rem;
            border-radius: 6px;
            max-width: 90%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
    <div class=\"search-grid\">
        {''.join(cards)}
    </div>
    """


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
        caption_output = gr.HTML()
        btn = gr.Button("Generate Captions") 
        btn.click(
            fn=batch_caption,
            inputs=gallery_input,
            outputs=caption_output
        )

    with gr.Tab("Text → Image"):
        text_input = gr.Textbox(label="Enter description")
        k = gr.Slider(1, 10, value=3, step=1, label="Number of images")
        btn2 = gr.Button("Search Images")
        gallery = gr.HTML()
        btn2.click(fn=text_to_images,
                   inputs=[text_input, k],
                   outputs=gallery, show_progress=True)

demo.launch()