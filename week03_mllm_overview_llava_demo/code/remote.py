from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image
import io
import torch

from transformers import LlavaForConditionalGeneration, AutoProcessor

"""Lightweight FastAPI app that exposes LLaVA image-question answering over HTTP."""

app = FastAPI()

# 允许任意来源、方法的跨域请求，方便本地或浏览器前端直接访问。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 加载模型并缓存 processor，便于后续 API 调用可以直接复用
print("加载 LLaVA 模型中...")

model_id = "models/llava-1.5-7b-hf"
# 运行时优先使用 GPU，否则回退 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

print("模型加载完成")

# 仅保留历史提示（目前未在逻辑中使用，方便以后扩展多轮上下文）
chat_history = []


@app.post("/api/ask")
async def ask(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """接受表单字段 question 和可选图像，返回 LLaVA 的回答."""
    global chat_history

    if image is None:
        # LLaVA 需要图像上下文才能工作，提前阻断无图请求
        return {"answer": "请上传图片（LLaVA 必须有图）"}

    contents = await image.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 👉 构造 prompt：LLaVA 用 <image> 占位符标记图像输入，后接提问、ASSISTANT 期待回复
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    # 模型推理阶段，限制生成令牌数避免过长而失控
    output = model.generate(
        **inputs,
        max_new_tokens=200
    )

    answer = processor.decode(output[0], skip_special_tokens=True)

    # 👉 去除 prompt 前缀，只保留模型实际生成的回答
    answer = answer.split("ASSISTANT:")[-1].strip()
    print(f"问题: {question}\n回答: {answer}\n")

    return {"answer": answer}
