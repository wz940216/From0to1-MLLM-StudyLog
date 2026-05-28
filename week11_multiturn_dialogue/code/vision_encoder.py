from PIL import Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel


def resolve_device(device):
    """根据配置选择真实可用设备，避免没有 GPU 时写死 cuda 导致程序直接崩溃。"""
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


class VisionEncoder(torch.nn.Module):
    """MiniLLaVA 的视觉编码器。

    这里使用 CLIP ViT 提取图像 patch 特征。MiniLLaVA 不把图片转成离散 token，
    而是将连续视觉特征经过 projector 映射到 LLM 的 embedding 空间，再和文本
    embedding 拼接后送入因果语言模型。
    """

    def __init__(self, model_path, freeze=True, device="cuda"):
        super().__init__()
        self.device = resolve_device(device)
        self.freeze = freeze

        # CLIPImageProcessor 负责 resize、center crop、normalize 等 CLIP 所需预处理。
        self.processor = CLIPImageProcessor.from_pretrained(model_path)
        self.vision_model = CLIPVisionModel.from_pretrained(model_path).to(self.device)

        # 预训练视觉塔通常冻结，只训练 projector 和语言模型；小数据集上这样更稳定。
        if freeze:
            self.vision_model.eval()
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        """将 PIL 图片列表编码为 patch 级视觉特征。

        Args:
            images: List[PIL.Image]，长度为 batch size。

        Returns:
            Tensor，形状为 (B, N, D)。以 CLIP ViT-B/16 为例，224x224 图片会得到
            14x14=196 个 patch，每个 patch 维度为 768。
        """
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        # 冻结视觉塔时关闭梯度，节省显存；不冻结时保留梯度用于端到端微调。
        if self.freeze:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
        else:
            outputs = self.vision_model(pixel_values=pixel_values)

        # last_hidden_state 形状为 (B, 1 + patch_num, hidden_dim)，第 0 个是 CLS。
        # CLS 更偏全局摘要，MiniLLaVA 需要给 LLM 更细粒度的图像信息，因此去掉 CLS，
        # 只保留 patch token 作为视觉上下文。
        patch_features = outputs.last_hidden_state[:, 1:, :]
        return patch_features


if __name__ == "__main__":
    vision_encoder = VisionEncoder(model_path="models/clip-vit-base-patch16", device="cuda")
    image_file = "dataset/coco128/images/train2017/000000000009.jpg"
    image = Image.open(image_file).convert("RGB")
    features = vision_encoder([image])
    print(features.shape)
