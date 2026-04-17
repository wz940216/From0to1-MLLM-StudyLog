from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


def generate_caption(pil_image=None, 
                     model_path="models/blip-image-captioning-base", 
                     max_length=50, num_beams=5, temperature=1.0):
    # ==============================
    # 1. 加载模型和预处理器
    # ==============================

    # BlipProcessor：负责把图片转换成模型可以理解的输入格式（如 tensor）
    processor = BlipProcessor.from_pretrained(model_path)

    # BlipForConditionalGeneration：BLIP 图像描述生成模型
    model = BlipForConditionalGeneration.from_pretrained(model_path)


    # ==============================
    # 2. 加载图片
    # ==============================

    # 这里使用本地图片路径（也可以是网络图片）
    if pil_image == None:
        print("No image provided, using default image.")
        img_url = "week05_blip_blip2_caption/code/000000039769.jpg"
        image = Image.open(img_url).convert("RGB")
    else:
        # 使用 PIL 打开图片，并转换为 RGB 格式（模型要求）
        image = pil_image[0].convert("RGB")

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
        max_length=max_length,     # 生成文本的最大长度
        num_beams=num_beams,       # Beam Search
        temperature=temperature    # 控制随机性（1.0为默认，越小越保守）
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
    return caption


# 封装成一个类，方便在 Web Demo 中调用
class BLIPCaptioner:
    def __init__(self, model_path="models/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_caption(self, pil_image, max_length=50, num_beams=5, temperature=1.0):
        image = pil_image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature
        )
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


if __name__ == "__main__":
    generate_caption()