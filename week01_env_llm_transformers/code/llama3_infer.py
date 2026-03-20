import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "models/Llama-3.1-8B-Instruct"

def main():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers in Chinese."},
        {"role": "user", "content": "用几句话说明大语言模型是怎么一步步生成答案的。"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(text)



if __name__ == "__main__":
    main()
