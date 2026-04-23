import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMDecoder(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

