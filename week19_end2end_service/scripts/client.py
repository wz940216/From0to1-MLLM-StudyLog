"""Client for calling the week19 /chat endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the MiniLLaVA FastAPI /chat endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:9000/chat")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="请描述这张图片。")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    data = {
        "question": args.question,
        "max_tokens": str(args.max_tokens),
        "temperature": str(args.temperature),
    }
    if args.system_prompt:
        data["system_prompt"] = args.system_prompt

    with image_path.open("rb") as f:
        response = requests.post(args.url, data=data, files={"image": (image_path.name, f)}, timeout=180)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

