"""Command line client for the week22 FastAPI service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the week22 /api/ask endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:9100/api/ask")
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--use-default-pdf", action="store_true")
    parser.add_argument("--question", default="根据这篇文章，总结文章创新点")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--enable-ppstructure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = {
        "question": args.question,
        "use_default_pdf": str(args.use_default_pdf).lower(),
        "max_images": str(args.max_images),
        "enable_ppstructure": str(args.enable_ppstructure).lower(),
    }
    if args.max_new_tokens is not None:
        data["max_new_tokens"] = str(args.max_new_tokens)

    files = None
    handle = None
    if args.pdf:
        pdf_path = Path(args.pdf)
        handle = pdf_path.open("rb")
        files = {"file": (pdf_path.name, handle, "application/pdf")}
    try:
        response = requests.post(args.url, data=data, files=files, timeout=1800)
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    main()

