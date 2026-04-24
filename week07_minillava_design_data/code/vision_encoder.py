from PIL import Image
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

class VisionEncoder(torch.nn.Module):
    def __init__(self, model_path, freeze=True, device="cuda"):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_path).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_path)
        self.device = device
        self.freeze = freeze
        # 冻结vision encoder参数，只训练projector和llm decoder的参数
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        if self.freeze:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
        else:
            outputs = self.vision_model(pixel_values=pixel_values)

        # (B, 197, 768)
        # 这里取clip中vision encoder的last_hidden_state输出，并且去掉cls的整体patch特征输出，只保留图像patch的特征输出
        # outputs中还有一个pooler_output是将cls token的输出进行layernorm之后的结果，代表整张图的全局特征，不适合用在理解图像
        # 反而patch中的特征更符合llm理解。
        last_hidden_state = outputs.last_hidden_state

        # 去掉 CLS → (B, 196, 768)
        patch_features = last_hidden_state[:, 1:, :]

        return patch_features
    
    
    
if __name__ == "__main__":
    
    vision_encoder = VisionEncoder(model_path="models/clip-vit-base-patch16", device="cuda")
    images_file = "dataset/coco128/images/train2017/000000000009.jpg"
    image = Image.open(images_file).convert("RGB")
    features = vision_encoder([image])
    print(features.shape)
