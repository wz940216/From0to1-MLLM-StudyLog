import torch
from transformers import CLIPModel, CLIPProcessor


class VisionEncoder(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def forward(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    
    
    
