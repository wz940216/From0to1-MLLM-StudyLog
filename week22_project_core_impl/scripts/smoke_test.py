"""Smoke test for the week22 alpha pipeline.

By default this validates PDF splitting and OCR on docs/notes/BLIP.pdf.
Use --run-vlm to include Qwen3-VL generation.
"""

from __future__ import annotations

import argparse

from week22_project_core_impl.core.config import Settings
from week22_project_core_impl.core.pipeline import DocumentQAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run week22 smoke test.")
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    pipeline = DocumentQAPipeline(settings)
    if args.run_vlm:
        result = pipeline.ask(
            settings.default_pdf,
            "根据这篇文章，总结文章创新点",
            max_images=args.max_images,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"request_id={result['request_id']}")
        print(f"selected_pages={result['selected_pages']}")
        print(result["answer"])
        return

    artifacts = pipeline.parse_pdf(settings.default_pdf)
    assert artifacts.pages, "expected rendered PDF pages"
    assert artifacts.ocr_pages, "expected OCR pages"
    assert artifacts.ocr_json_path.exists(), "expected ocr.json"
    print(f"ok request_id={artifacts.request_id} pages={len(artifacts.pages)} ocr_json={artifacts.ocr_json_path}")


if __name__ == "__main__":
    main()

