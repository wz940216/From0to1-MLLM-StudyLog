import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft import LoraConfig, TaskType

MODEL_NAME = "/root/zheng/code/mllm/Qwen1.5-1.8B-Chat"

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
print("原始模型结构")
print(model)
model = get_peft_model(model, peft_config)
print("LORA 模型结构")
print(model)
print("参数统计")
model.print_trainable_parameters()

