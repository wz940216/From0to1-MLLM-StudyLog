import argparse
import json
import os

from PIL import Image
import torch

from mini_llava import MiniLlavaModel


IMAGE_TOKEN = "<image>"
JSON_OUTPUT_PROMPT = """请只输出一个合法 JSON 对象，不要输出 markdown 代码块或额外解释。
JSON 格式固定为：
{"answer": "这里填写对用户问题的回答"}"""


def load_checkpoint(model, checkpoint_path):
    """加载训练得到的 checkpoint。

    当前 train.py 保存的是一个字典，其中 "model" 只包含可训练参数和 LoRA adapter。
    因此这里使用 strict=False，让冻结的基础模型参数继续来自初始化权重。
    """
    if checkpoint_path is None:
        return
    state = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(state["model"], strict=False)


def clean_user_text(text):
    """推理时用户只输入自然语言问题，避免手动输入多个 <image>。"""
    return str(text).replace(IMAGE_TOKEN, "").strip()


def add_json_output_prompt(question):
    """给当前轮问题追加固定输出格式约束。"""
    question = clean_user_text(question)
    if not question:
        return JSON_OUTPUT_PROMPT
    return f"{question}\n\n{JSON_OUTPUT_PROMPT}"


def build_context_prompt(history, question):
    """把历史对话和当前问题拼成 MiniLLaVA 可用的多轮 prompt。

    MiniLLaVA 的底层模型仍然依赖 <image> token 来确定视觉特征插入位置，
    所以整段 prompt 必须满足：
    - 只有第一轮 USER 含有 <image>。
    - 历史 assistant 回答保留在上下文里。
    - 当前轮以 "ASSISTANT: " 结尾，让模型继续生成本轮回答。

    history 的格式是：
    [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]
    """
    pieces = []
    image_token_used = False

    for message in history:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            content = clean_user_text(content)
            if not image_token_used:
                pieces.append(f"USER: {IMAGE_TOKEN}\n{content}\n")
                image_token_used = True
            else:
                pieces.append(f"USER: {content}\n")
        elif role == "assistant":
            pieces.append(f"ASSISTANT: {content}\n")

    question = add_json_output_prompt(question)
    if not image_token_used:
        pieces.append(f"USER: {IMAGE_TOKEN}\n{question}\nASSISTANT: ")
    else:
        pieces.append(f"USER: {question}\nASSISTANT: ")

    return "".join(pieces)


def normalize_generated_text(text, prompt):
    """尽量只保留当前轮 assistant 的回答。

    不同 Transformers 版本在使用 inputs_embeds 做 generate 时，decode 出来的内容
    可能是纯新增 token，也可能包含一部分 prompt。这里做轻量清理：
    - 如果输出以完整 prompt 开头，去掉 prompt。
    - 如果模型继续生成了下一轮 USER，把下一轮之前的内容作为本轮回答。
    """
    text = str(text).strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    if "USER:" in text:
        text = text.split("USER:", 1)[0].strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidate = text[start:end + 1]
        try:
            json.loads(candidate)
            text = candidate
        except json.JSONDecodeError:
            pass
    return text.strip()


def build_openai_messages(image_path, history):
    """把内部上下文导出成用户给出的 OpenAI chat messages 形式。"""
    messages = []
    for idx, message in enumerate(history):
        role = message["role"]
        content = message["content"]
        if idx == 0 and role == "user":
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": content},
                ],
            })
        else:
            messages.append({"role": role, "content": content})
    return messages


def load_context(context_file):
    """从 OpenAI messages JSON 中恢复历史上下文。"""
    if context_file is None or not os.path.exists(context_file):
        return []

    with open(context_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages", data if isinstance(data, list) else [])

    history = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user" and isinstance(content, list):
            text_parts = [
                str(block.get("text", "")).strip()
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "\n".join(part for part in text_parts if part)
        if role in {"user", "assistant"} and str(content).strip():
            history.append({"role": role, "content": str(content).strip()})
    return history


def save_context(context_file, image_path, history):
    """把当前上下文保存为 OpenAI chat messages JSON，方便下次继续对话。"""
    if context_file is None:
        return
    parent = os.path.dirname(context_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump({"messages": build_openai_messages(image_path, history)}, f, ensure_ascii=False, indent=2)
        f.write("\n")


def answer_one_turn(model, image, history, question, gen_config):
    """执行一轮带上下文的图文对话推理，并返回 assistant 回答。"""
    prompt = build_context_prompt(history, question)
    outputs = model.generate(
        images=[image],
        prompts=[prompt],
        max_new_tokens=int(gen_config["MAX_NEW_TOKENS"]),
        temperature=float(gen_config["TEMPERATURE"]),
        do_sample=bool(gen_config["DO_SAMPLE"]),
        top_p=float(gen_config["TOP_P"]),
        top_k=int(gen_config["TOP_K"]),
        repetition_penalty=float(gen_config["REPETITION_PENALTY"]),
    )
    return normalize_generated_text(outputs[0], prompt)


def main():
    parser = argparse.ArgumentParser(description="MiniLLaVA 多轮图文对话推理脚本")
    parser.add_argument("--config", default="week11_multiturn_dialogue/code/config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="week11_multiturn_dialogue/outputs/checkpoints/step_2109.pt",
        help="训练得到的 .pt 检查点路径；传入 none 可跳过加载。",
    )
    parser.add_argument("--image", default="dataset/coco128/images/train2017/000000000025.jpg", help="输入图片路径")
    parser.add_argument(
        "--question",
        action="append",
        default=None,
        help="单轮问题；可重复传入多次，脚本会按顺序保留上下文。",
    )
    parser.add_argument("--interactive", action="store_true", help="进入交互式多轮对话。")
    parser.add_argument("--context-file", default=None, help="可选：读取/保存 OpenAI messages 格式上下文 JSON。")
    args = parser.parse_args()

    checkpoint = None if str(args.checkpoint).lower() in {"none", "null", ""} else args.checkpoint

    model = MiniLlavaModel(args.config)
    load_checkpoint(model, checkpoint)

    image = Image.open(args.image).convert("RGB")
    gen_config = model.config["INFERENCE"]["GENERATION"]
    history = load_context(args.context_file)

    if args.interactive:
        print("进入多轮对话。输入 exit/quit/q 结束，输入 clear 清空上下文。")
        while True:
            question = input("USER: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            if question.lower() == "clear":
                history = []
                save_context(args.context_file, args.image, history)
                print("上下文已清空。")
                continue
            if not question:
                continue

            answer = answer_one_turn(model, image, history, question, gen_config)
            history.extend([
                {"role": "user", "content": clean_user_text(question)},
                {"role": "assistant", "content": answer},
            ])
            save_context(args.context_file, args.image, history)
            print(f"ASSISTANT: {answer}")
        return

    questions = args.question or ["Please describe this image"]
    for question in questions:
        answer = answer_one_turn(model, image, history, question, gen_config)
        history.extend([
            {"role": "user", "content": clean_user_text(question)},
            {"role": "assistant", "content": answer},
        ])
        print(f"USER: {question}")
        print(f"ASSISTANT: {answer}")

    save_context(args.context_file, args.image, history)


if __name__ == "__main__":
    main()
