from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# 基础模型路径（Qwen1.5 1.8B Chat）
base_model_name = "models/Qwen1.5-1.8B-Chat"
# LoRA 训练检查点目录（已训练好的 LoRA 权重）
lora_model_dir  = "week2/Qwen1.5-1.8B-Chat-lora/checkpoint-2000"   # 训练好的 LoRA 保存目录
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# 加载基础因果语言模型
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
# 加载 LoRA 权重到基础模型上
model = PeftModel.from_pretrained(base_model, lora_model_dir)
# 设置推理模式，并搬运到 GPU
model.eval().to("cuda")
# 准备输入指令文本
text = "给出保持健康的三个建议。"
prompt = f"指令: {text}\n输入: \n回复: "
# 将 prompt token 化为张量
inputs = tokenizer(prompt, return_tensors="pt")
# 用 LoRA 模型生成
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=250)
# 丢弃 prompt 部分，只取模型新生成的 token
gen_ids = outputs[0][inputs["input_ids"].shape[-1]:].detach().cpu().numpy()
# 将 token id 解码为文本（跳过特殊 token）
response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print("lora model response:")
print(response[0])
# 用基础模型生成（对照结果）
outputs = base_model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=250)
gen_ids = outputs[0][inputs["input_ids"].shape[-1]:].detach().cpu().numpy()
response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
print("base model response:")
print(response[0])

