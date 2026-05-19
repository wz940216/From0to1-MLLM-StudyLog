import argparse
import json
from pathlib import Path


CHINESE_PROMPT = "<image>\n请描述这张图片。"
ENGLISH_PROMPT = "<image>\nPlease describe this image."


def has_chinese(text: str) -> bool:
    """Return True when the caption contains CJK unified ideographs."""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def build_image_map(coco_data: dict) -> dict[int, str]:
    return {image["id"]: image["file_name"] for image in coco_data.get("images", [])}


def convert_caption_file(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    image_id_to_file = build_image_map(coco_data)
    qa_data = []

    for ann in coco_data.get("annotations", []):
        caption = ann["caption"].strip()
        image_id = ann["image_id"]
        image_file = image_id_to_file.get(image_id, f"{image_id:012d}.jpg")
        prompt = CHINESE_PROMPT if has_chinese(caption) else ENGLISH_PROMPT

        qa_data.append(
            {
                "id": str(ann["id"]).zfill(7),
                "image": image_file,
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt,
                    },
                    {
                        "from": "gpt",
                        "value": caption,
                    },
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    return len(qa_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO caption annotations to single-turn image QA data."
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("dataset/COCOCaption/annotations"),
        help="Directory containing captions_train2017.json and captions_val2017.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/COCOCaption/annotations"),
        help="Directory to write converted QA json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ("train2017", "val2017")

    for split in splits:
        input_path = args.annotation_dir / f"captions_{split}.json"
        output_path = args.output_dir / f"captions_{split}_qa.json"
        count = convert_caption_file(input_path, output_path)
        print(f"Wrote {count} QA samples to {output_path}")


if __name__ == "__main__":
    main()
