import argparse
import json
import os
import sys
from datetime import datetime

from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from infer import answer_one_turn, clean_user_text, load_checkpoint, truncate_history
from mini_llava import MiniLlavaModel
from output_parser import parse_json_output


DEFAULT_DIALOGUE = [
    "请描述这张图片的主要内容。",
    "画面里最重要的物体是什么？请用 JSON 回答。",
    "继续保持 JSON，说明你判断的依据。",
    "如果只保留最近上下文，你还能回答刚才的问题吗？请用 JSON 回答。",
]


def append_jsonl(path, record):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="固定多轮对话自动测试脚本")
    parser.add_argument("--config", default="week14_dialogue_stability_output_control/configs/config.yaml")
    parser.add_argument("--checkpoint", default="none", help="训练得到的 .pt；none 表示只跑初始化模型流程。")
    parser.add_argument("--image", default="dataset/coco128/images/train2017/000000000025.jpg")
    parser.add_argument("--output", default="week14_dialogue_stability_output_control/outputs/logs/multiturn_test.jsonl")
    parser.add_argument("--max-history-turns", type=int, default=3)
    parser.add_argument("--question", action="append", default=None, help="覆盖默认固定对话，可重复传入。")
    args = parser.parse_args()

    model = MiniLlavaModel(args.config)
    checkpoint = None if str(args.checkpoint).lower() in {"none", "null", ""} else args.checkpoint
    load_checkpoint(model, checkpoint)

    image = Image.open(args.image).convert("RGB")
    gen_config = model.config["INFERENCE"]["GENERATION"]
    history = []
    questions = args.question or DEFAULT_DIALOGUE

    for turn_idx, question in enumerate(questions, start=1):
        answer = answer_one_turn(
            model,
            image,
            history,
            question,
            gen_config,
            max_history_turns=args.max_history_turns,
        )
        history.extend([
            {"role": "user", "content": clean_user_text(question)},
            {"role": "assistant", "content": answer},
        ])
        history = truncate_history(history, args.max_history_turns)

        parsed = parse_json_output(answer, required_keys=["answer"])
        append_jsonl(args.output, {
            "event": "multiturn_test",
            "time": datetime.now().isoformat(timespec="seconds"),
            "turn": turn_idx,
            "image": args.image,
            "question": question,
            "answer": answer,
            "json_ok": parsed.ok,
            "json_error": parsed.error,
            "history_turns_after_truncate": len(history) // 2,
        })
        print(f"[{turn_idx}] USER: {question}")
        print(f"[{turn_idx}] ASSISTANT: {answer}")


if __name__ == "__main__":
    main()
