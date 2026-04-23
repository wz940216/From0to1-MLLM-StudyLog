import torch
from vision_encoder import VisionEncoder
from llm_decoder import LLMDecoder

class MiniLlavaModel(torch.nn.Module):
    def __init__(self):
        super(MiniLlavaModel, self).__init__()
        self.vision_encoder = VisionEncoder(model_path="models/clip-vit-base-patch16")
        self.language_decoder = LLMDecoder(model_path="models/Qwen1.5-1.8B")
        self.projector = torch.nn.ModuleList([torch.nn.Linear(512, 4096),
                                              torch.nn.GELU(),
                                              torch.nn.Linear(4096, 4096)])  # Adjust dimensions as needed

    def forward(self, images, input_ids, attention_mask):
        pass