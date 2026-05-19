import argparse
import json
from pathlib import Path


DEFAULT_ANNOTATIONS = Path("dataset/VQA/abstract_v002_val2017_annotations.json")
DEFAULT_QUESTIONS = Path("dataset/VQA/OpenEnded_abstract_v002_val2017_questions.json")
DEFAULT_OUTPUT = Path("dataset/VQA/abstract_v002_val2017_qa.json")
DEFAULT_IMAGE_PREFIX = "abstract_v002_val2015"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_id(value):
    return str(value).zfill(7)


def format_image_name(image_id, image_prefix=DEFAULT_IMAGE_PREFIX):
    return f"{image_prefix}_{int(image_id):012d}.png"


def convert(annotations_path, questions_path, output_path, image_prefix, image_root):
    annotations_data = load_json(annotations_path)
    questions_data = load_json(questions_path)

    annotations = annotations_data["annotations"]
    questions = questions_data["questions"]
    question_by_id = {item["question_id"]: item for item in questions}

    qa_data = []
    missing_question_ids = 0

    for ann in annotations:
        question_id = ann["question_id"]
        question_item = question_by_id.get(question_id)
        if question_item is None:
            missing_question_ids += 1
            continue

        image_id = ann.get("image_id", question_item["image_id"])
        image_name = format_image_name(image_id, image_prefix)
        if image_root:
            image_name = str(Path(image_root) / image_name)

        qa_data.append(
            {
                "id": format_id(question_id),
                "image": image_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n" + question_item["question"].strip(),
                    },
                    {
                        "from": "gpt",
                        "value": ann["multiple_choice_answer"].strip(),
                    },
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(qa_data)} samples to {output_path}")
    if missing_question_ids:
        print(f"Skipped {missing_question_ids} annotations without matched questions")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VQA abstract validation annotations/questions to QA conversation format."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help="Image filename prefix. The provided val2017 abstract images use abstract_v002_val2015 names.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image directory prefix to include in the output image field.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    convert(
        annotations_path=args.annotations,
        questions_path=args.questions,
        output_path=args.output,
        image_prefix=args.image_prefix,
        image_root=args.image_root,
    )


if __name__ == "__main__":
    main()
