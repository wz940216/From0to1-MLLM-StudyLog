import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 这里是要加载的模型名称（Hub 里模型全名）
MODEL_NAME = "models/Qwen1.5-1.8B-Chat"


def main():
    """脚本入口：加载 tokenizer 和模型，构建聊天输入，生成并输出回答。"""

    # 1) 加载 tokenizer
    # trust_remote_code=True：允许从模型 repo 里加载自定义 tokenizer 配置
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    # 2) 加载预训练模型
    # torch_dtype=torch.bfloat16：用 bf16 精度，节省显存并加速推理
    # device_map="auto"：自动把模型层分配到可用 GPU（如多卡）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 3) 构造消息列表（聊天上下文）。
    #   role=system 是助手人格或行为说明，role=user 是用户输入。
    messages = [
        {"role": "system", "content": "你是一个乐于助人的中文助手。"},
        {"role": "user", "content": "简单介绍一下大语言模型是如何进行推理的。"}
    ]

    # 4) 应用聊天模板，组合成 raw prompt 文本
    # tokenize=False：不直接返回 token，否则我们后面再用 tokenizer() 处理
    # add_generation_prompt=True：加上生成提示文本（如助手回答开头）
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 5) tokenizer 转 tensor，并移动到模型所在设备
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 6) 关闭梯度计算，加速并节省显存
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,      # 生成的最大 token 数
            do_sample=True,          # 采样生成而不是贪心解码
            top_p=0.9,               # nucleus sampling
            temperature=0.7,         # 采样温度
        )

    # 7) decode 只拿新生成 token，去掉输入 prompt 部分
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    # skip_special_tokens=True：去掉特殊符号（<s>, </s> 等）
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 8) 打印最后结果文本
    print(text)


if __name__ == "__main__":
    main()
