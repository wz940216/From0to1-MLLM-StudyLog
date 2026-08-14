"""Local CLI for running PDF parsing and optional Qwen3-VL QA without FastAPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from week22_project_core_impl.core.config import Settings
from week22_project_core_impl.core.pipeline import DocumentQAPipeline


def parse_args() -> argparse.Namespace:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Run the week22 document QA pipeline.")
    parser.add_argument("--pdf", type=Path, default=settings.default_pdf)
    parser.add_argument("--question", default="根据这篇文章，总结文章创新点")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--enable-ppstructure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    pipeline = DocumentQAPipeline(settings)
    if args.parse_only:
        artifacts = pipeline.parse_pdf(args.pdf, enable_ppstructure=args.enable_ppstructure)
        print(json.dumps(artifacts.to_json(), ensure_ascii=False, indent=2))
        return
    result = pipeline.ask(
        args.pdf,
        args.question,
        max_images=args.max_images,
        max_new_tokens=args.max_new_tokens,
        enable_ppstructure=args.enable_ppstructure,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

