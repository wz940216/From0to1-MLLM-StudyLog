"""Smoke test for the week23 beta pipeline.

Default mode validates document hash cache, PDF splitting and OCR on docs/notes/BLIP.pdf.
Use --run-vlm to include Qwen3-VL generation.
"""

from __future__ import annotations

import argparse

from week23_project_optimize.core.config import Settings
from week23_project_optimize.core.pipeline import DocumentQAPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run week23 smoke test.")
    parser.add_argument("--run-vlm", action="store_true")
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--use-answer-cache", action="store_true")
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
            prompt_type="summary",
            use_answer_cache=args.use_answer_cache,
        )
        print(f"request_id={result['request_id']}")
        print(f"document_cache_hit={result['document_cache_hit']} answer_cache_hit={result['answer_cache_hit']}")
        print(f"selected_pages={result['selected_pages']} prompt_type={result['prompt_type']}")
        print(result["answer"])
        return

    first = pipeline.parse_pdf(settings.default_pdf)
    first_meta = dict(pipeline.last_parse_meta)
    second = pipeline.parse_pdf(settings.default_pdf)
    second_meta = dict(pipeline.last_parse_meta)
    assert first.pages, "expected rendered PDF pages"
    assert first.ocr_pages, "expected OCR pages"
    assert first.ocr_json_path.exists(), "expected ocr.json"
    assert second_meta["cache_hit"], "expected second parse to hit document cache"
    print(
        "ok "
        f"document_hash={first_meta['document_hash']} "
        f"pages={len(second.pages)} "
        f"first_cache_hit={first_meta['cache_hit']} "
        f"second_cache_hit={second_meta['cache_hit']} "
        f"ocr_json={second.ocr_json_path}"
    )


if __name__ == "__main__":
    main()
