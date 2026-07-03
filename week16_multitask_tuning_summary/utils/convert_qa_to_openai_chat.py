"""
使用方法
python week16_multitask_tuning_summary/utils/convert_qa_to_openai_chat.py \
  --input dataset/LLaVA-Instruct-150K/llava_instruct_150k.json \
  --output dataset/LLaVA-Instruct-150K/llava_instruct_150k_openai.json \
  --image-prefix train2017 \
  --preserve-conversations
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

CAPTION_FOLLOWUPS = [
    "Can you describe the image in another way?",
    "Give another concise caption for this image.",
    "What is another valid description of the picture?",
    "Describe the same image with different wording.",
]

DEFAULT_SYSTEM_MESSAGES = [
    "You are a helpful visual assistant. Answer the user's questions based on the image.",
    "You are a concise visual assistant. Use the image to answer the user's question clearly.",
    "You are a careful multimodal assistant. Ground your answer in the visual content.",
    "You are an image understanding assistant. Respond to the user based only on the image and the conversation.",
]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def remove_image_token(text):
    """OpenAI chat 格式把图片放进 content 的 image block，文本里不再需要 <image>。"""
    text = str(text).replace("<image>", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def join_image_ref(image_prefix, image_name):
    """构造写入 OpenAI messages 的图片路径。

    输出 JSON 里的 image 字段应该能在训练时定位到真实图片。这里使用相对
    dataset root 的路径，例如：
    - COCO: val2017/000000179765.jpg
    - VQA: scene_img_abstract_v002_val2017/xxx.png
    - CC3M: images/GCC_train_xxx.jpg
    """
    image_name = str(image_name).strip()
    if not image_prefix:
        return image_name.replace("\\", "/")
    return os.path.join(image_prefix, image_name).replace("\\", "/")


def extract_first_qa(item):
    """从旧 LLaVA conversations 样本中取出第一组 human/gpt QA。"""
    question = None
    answer = None
    for message in item.get("conversations", []):
        role = message.get("from")
        value = str(message.get("value", "")).strip()
        if role == "human" and question is None:
            question = value
        elif role == "gpt" and answer is None:
            answer = value
        if question is not None and answer is not None:
            break

    if not question or not answer or "image" not in item:
        return None

    return {
        "id": str(item.get("id", "")),
        "image": item["image"],
        "question": question,
        "answer": answer,
    }


def maybe_rewrite_repeated_question(question, seen_questions, turn_index):
    """caption 数据常常多条样本问题完全相同，这里改成自然的追问形式。"""
    key = remove_image_token(question).lower()
    if key not in seen_questions:
        seen_questions.add(key)
        return question
    return CAPTION_FOLLOWUPS[(turn_index - 1) % len(CAPTION_FOLLOWUPS)]


def append_user_message(messages, question, image_ref=None):
    """追加 OpenAI chat user 消息。

    第一轮 user 消息使用多模态 content：image block + text block。
    后续轮次只包含文本，符合用户给出的示例格式。
    """
    question = remove_image_token(question)
    if image_ref is None:
        messages.append({
            "role": "user",
            "content": question,
        })
        return

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_ref,
            },
            {
                "type": "text",
                "text": question,
            },
        ],
    })


def append_system_message(messages, system_message):
    messages.append({
        "role": "system",
        "content": str(system_message).strip(),
    })


def append_assistant_message(messages, answer):
    messages.append({
        "role": "assistant",
        "content": str(answer).strip(),
    })



def build_preserved_conversation_samples(items, image_prefix, system_message, seed):
    """逐条保留 LLaVA conversations 的所有轮次，直接转成 OpenAI chat messages。"""
    rng = random.Random(seed)
    output = []
    skipped = 0
    dangling_messages = 0

    for item in items:
        image = item.get("image")
        conversations = item.get("conversations", [])
        if not image or not isinstance(conversations, list):
            skipped += 1
            continue

        messages = []
        selected_system_message = system_message or rng.choice(DEFAULT_SYSTEM_MESSAGES)
        append_system_message(messages, selected_system_message)
        image_ref = join_image_ref(image_prefix, image)
        image_attached = False
        user_turns = 0
        assistant_turns = 0
        pending_user = False

        for message in conversations:
            role = message.get("from")
            value = str(message.get("value", "")).strip()
            if not value:
                continue

            if role == "human":
                if pending_user:
                    dangling_messages += 1
                append_user_message(
                    messages,
                    question=value,
                    image_ref=image_ref if not image_attached else None,
                )
                image_attached = True
                user_turns += 1
                pending_user = True
            elif role == "gpt":
                if not pending_user:
                    dangling_messages += 1
                append_assistant_message(messages, value)
                assistant_turns += 1
                pending_user = False

        if user_turns == 0 or assistant_turns == 0 or not image_attached:
            skipped += 1
            continue
        if pending_user:
            dangling_messages += 1

        sample_id = str(item.get("id") or os.path.splitext(os.path.basename(str(image)))[0])
        output.append({
            "id": sample_id,
            "messages": messages,
            "source_ids": [sample_id],
        })

    return output, {
        "input_samples": len(items),
        "output_samples": len(output),
        "skipped_invalid_samples": skipped,
        "dangling_messages": dangling_messages,
    }


def build_multiturn_samples(items, min_turns, max_turns, shuffle, seed, rewrite_repeated_questions, image_prefix, system_message):
    grouped = defaultdict(list)
    skipped = 0

    for item in items:
        qa = extract_first_qa(item)
        if qa is None:
            skipped += 1
            continue
        grouped[qa["image"]].append(qa)

    rng = random.Random(seed)
    output = []
    dropped_below_min = 0

    for image, qas in grouped.items():
        if shuffle:
            rng.shuffle(qas)

        for chunk_idx, start in enumerate(range(0, len(qas), max_turns)):
            chunk = qas[start:start + max_turns]
            if len(chunk) < min_turns:
                dropped_below_min += len(chunk)
                continue

            messages = []
            selected_system_message = system_message or rng.choice(DEFAULT_SYSTEM_MESSAGES)
            append_system_message(messages, selected_system_message)
            seen_questions = set()
            source_ids = []
            image_ref = join_image_ref(image_prefix, image)

            for turn_idx, qa in enumerate(chunk):
                question = qa["question"]
                if rewrite_repeated_questions:
                    question = maybe_rewrite_repeated_question(question, seen_questions, turn_idx)

                append_user_message(
                    messages,
                    question=question,
                    image_ref=image_ref if turn_idx == 0 else None,
                )
                append_assistant_message(messages, qa["answer"])
                if qa["id"]:
                    source_ids.append(qa["id"])

            sample_id = f"{os.path.splitext(os.path.basename(image))[0]}_{chunk_idx:03d}"
            output.append({
                "id": sample_id,
                "messages": messages,
                "source_ids": source_ids,
            })

    return output, {
        "input_samples": len(items),
        "image_groups": len(grouped),
        "output_samples": len(output),
        "skipped_invalid_samples": skipped,
        "dropped_turns_below_min_turns": dropped_below_min,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert single-turn LLaVA-style QA data into OpenAI multi-turn chat messages by grouping on image."
    )
    parser.add_argument("--input", required=True, help="Input LLaVA-style QA JSON.")
    parser.add_argument("--output", required=True, help="Output OpenAI chat JSON.")
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Relative image directory written into the first user image block, e.g. val2017 or images.",
    )
    parser.add_argument("--min-turns", type=int, default=2, help="Drop image groups/chunks with fewer turns.")
    parser.add_argument("--max-turns", type=int, default=4, help="Maximum QA turns per output conversation.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle QA order within each image group.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-single-turn",
        action="store_true",
        help="Keep images that have only one QA as one-turn conversations.",
    )
    parser.add_argument(
        "--no-rewrite-repeated-questions",
        action="store_true",
        help="Do not rewrite repeated caption prompts into follow-up questions.",
    )
    parser.add_argument(
        "--preserve-conversations",
        action="store_true",
        help="Preserve every original LLaVA conversations turn instead of regrouping only first QA pairs by image.",
    )
    parser.add_argument(
        "--system-message",
        default="",
        help="Fixed system message inserted as the first message. If omitted, one default system message is selected randomly per conversation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_turns < 1:
        raise ValueError("--max-turns must be >= 1")

    min_turns = 1 if args.keep_single_turn else args.min_turns
    if min_turns < 1:
        raise ValueError("--min-turns must be >= 1")
    if min_turns > args.max_turns:
        raise ValueError("--min-turns cannot be greater than --max-turns")

    items = read_json(args.input)
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of LLaVA-style samples.")

    if args.preserve_conversations:
        output, stats = build_preserved_conversation_samples(
            items=items,
            image_prefix=args.image_prefix,
            system_message=args.system_message,
            seed=args.seed,
        )
    else:
        output, stats = build_multiturn_samples(
            items=items,
            min_turns=min_turns,
            max_turns=args.max_turns,
            shuffle=args.shuffle,
            seed=args.seed,
            rewrite_repeated_questions=not args.no_rewrite_repeated_questions,
            image_prefix=args.image_prefix,
            system_message=args.system_message,
        )
    write_json(output, args.output)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
