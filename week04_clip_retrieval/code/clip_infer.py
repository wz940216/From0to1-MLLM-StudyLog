#https://huggingface.co/docs/transformers/model_doc/clip?usage=AutoModel
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("models/clip-vit-base-patch16", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

url = "week04_clip_retrieval/code/000000039769.jpg"
image = Image.open(url)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = labels[most_likely_idx]
print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")