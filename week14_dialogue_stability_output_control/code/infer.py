import argparse
import json
import os
from datetime import datetime

from PIL import Image
import torch

from mini_llava import MiniLlavaModel
from output_parser import JSON_REPAIR_PROMPT, normalize_model_text, parse_json_output
from prompt_templates import IMAGE_TOKEN, build_json_few_shot_question, clean_user_text


def append_jsonl(path, record):
    """向 JSONL 推理日志追加一条记录。"""
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_checkpoint(model, checkpoint_path):
    """加载训练得到的 checkpoint。

    当前 train.py 保存的是一个字典，其中 "model" 只包含可训练参数和 LoRA adapter。
    因此这里使用 strict=False，让冻结的基础模型参数继续来自初始化权重。
    """
    if checkpoint_path is None:
        return
    state = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(state["model"], strict=False)


def add_json_output_prompt(question):
    """给当前轮问题追加 few-shot JSON 输出模板。"""
    return build_json_few_shot_question(question)


def extract_text_content(content):
    """从 OpenAI chat content 中提取纯文本。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()

    text_parts = [
        str(block.get("text", "")).strip()
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part).strip()


def truncate_history(history, max_history_turns=None):
    """保留最近 N 轮 user/assistant 历史，降低长对话跑偏风险。"""
    if max_history_turns is None or int(max_history_turns) <= 0:
        return list(history)
    return list(history)[-int(max_history_turns) * 2:]


def build_context_prompt(history, question, system_prompt=None, max_history_turns=None):
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
    system_prompt = str(system_prompt or "").strip()
    if system_prompt:
        pieces.append(f"SYSTEM: {system_prompt}\n")

    image_token_used = False

    for message in truncate_history(history, max_history_turns=max_history_turns):
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


def build_openai_messages(image_path, history, system_prompt=None):
    """把内部上下文导出成用户给出的 OpenAI chat messages 形式。"""
    messages = []
    system_prompt = str(system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

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
        return [], ""

    with open(context_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("messages", data if isinstance(data, list) else [])

    history = []
    system_parts = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_text = extract_text_content(content)
            if system_text:
                system_parts.append(system_text)
            continue
        if role == "user" and isinstance(content, list):
            content = extract_text_content(content)
        if role in {"user", "assistant"} and str(content).strip():
            history.append({"role": role, "content": str(content).strip()})
    return history, "\n".join(system_parts).strip()


def save_context(context_file, image_path, history, system_prompt=None):
    """把当前上下文保存为 OpenAI chat messages JSON，方便下次继续对话。"""
    if context_file is None:
        return
    parent = os.path.dirname(context_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump(
            {"messages": build_openai_messages(image_path, history, system_prompt=system_prompt)},
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


def _generate_text(model, image, prompt, gen_config):
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
    return normalize_model_text(outputs[0], prompt)


def answer_one_turn(model, image, history, question, gen_config, system_prompt=None, max_history_turns=None, retry_on_json_error=True):
    """执行一轮带上下文的图文对话推理，并返回 assistant 回答。"""
    prompt = build_context_prompt(
        history,
        question,
        system_prompt=system_prompt,
        max_history_turns=max_history_turns,
    )
    answer = _generate_text(model, image, prompt, gen_config)
    parsed = parse_json_output(answer, required_keys=["answer"])
    if parsed.ok:
        return parsed.text
    if not retry_on_json_error:
        return answer

    retry_question = f"{clean_user_text(question)}\n\n{JSON_REPAIR_PROMPT}\n上一轮错误：{parsed.error}"
    retry_prompt = build_context_prompt(
        history,
        retry_question,
        system_prompt=system_prompt,
        max_history_turns=max_history_turns,
    )
    retry_answer = _generate_text(model, image, retry_prompt, gen_config)
    retry_parsed = parse_json_output(retry_answer, required_keys=["answer"])
    return retry_parsed.text if retry_parsed.ok else retry_answer


def main():
    parser = argparse.ArgumentParser(description="MiniLLaVA 多轮图文对话推理脚本")
    parser.add_argument("--config", default="week14_dialogue_stability_output_control/configs/config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="week14_dialogue_stability_output_control/outputs/checkpoints/step_2109.pt",
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
    parser.add_argument("--system", default=None, help="可选：system 指令；优先级高于 context-file 中的 system 消息。")
    parser.add_argument("--log-file", default=None, help="可选：推理 JSONL 日志路径；默认使用配置中的 INFERENCE.LOG_FILE。")
    parser.add_argument("--max-history-turns", type=int, default=3, help="上下文只保留最近 N 轮；<=0 表示不截断。")
    args = parser.parse_args()

    checkpoint = None if str(args.checkpoint).lower() in {"none", "null", ""} else args.checkpoint

    model = MiniLlavaModel(args.config)
    load_checkpoint(model, checkpoint)

    image = Image.open(args.image).convert("RGB")
    gen_config = model.config["INFERENCE"]["GENERATION"]
    log_file = args.log_file or model.config.get("INFERENCE", {}).get("LOG_FILE")
    history, context_system = load_context(args.context_file)
    system_prompt = str(args.system).strip() if args.system is not None else context_system

    if args.interactive:
        print("进入多轮对话。输入 exit/quit/q 结束，输入 clear 清空上下文。")
        while True:
            question = input("USER: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                break
            if question.lower() == "clear":
                history = []
                save_context(args.context_file, args.image, history, system_prompt=system_prompt)
                print("上下文已清空。")
                continue
            if not question:
                continue

            answer = answer_one_turn(model, image, history, question, gen_config, system_prompt=system_prompt, max_history_turns=args.max_history_turns)
            history.extend([
                {"role": "user", "content": clean_user_text(question)},
                {"role": "assistant", "content": answer},
            ])
            history = truncate_history(history, args.max_history_turns)
            save_context(args.context_file, args.image, history, system_prompt=system_prompt)
            append_jsonl(log_file, {
                "event": "infer_turn",
                "time": datetime.now().isoformat(timespec="seconds"),
                "image": args.image,
                "question": question,
                "answer": answer,
                "history_turns": len(history) // 2,
                "interactive": True,
            })
            print(f"ASSISTANT: {answer}")
        return

    questions = args.question or ["Please describe this image"]
    for question in questions:
        answer = answer_one_turn(model, image, history, question, gen_config, system_prompt=system_prompt, max_history_turns=args.max_history_turns)
        history.extend([
            {"role": "user", "content": clean_user_text(question)},
            {"role": "assistant", "content": answer},
        ])
        history = truncate_history(history, args.max_history_turns)
        append_jsonl(log_file, {
            "event": "infer_turn",
            "time": datetime.now().isoformat(timespec="seconds"),
            "image": args.image,
            "question": question,
            "answer": answer,
            "history_turns": len(history) // 2,
            "interactive": False,
        })
        print(f"USER: {question}")
        print(f"ASSISTANT: {answer}")

    save_context(args.context_file, args.image, history, system_prompt=system_prompt)


if __name__ == "__main__":
    main()
