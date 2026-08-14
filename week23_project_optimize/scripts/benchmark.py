"""Lightweight benchmark for the week23 FastAPI API.

Default target: concurrency 1/2/4, 5 requests each.
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark week23 /api/ask.")
    parser.add_argument("--url", default="http://127.0.0.1:9200/api/ask")
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--use-default-pdf", action="store_true")
    parser.add_argument("--question", default="根据这篇文章，总结文章创新点")
    parser.add_argument("--prompt-type", default="summary")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--requests-per-level", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--use-answer-cache", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


def call_once(args: argparse.Namespace) -> dict[str, Any]:
    data = {
        "question": args.question,
        "use_default_pdf": str(args.use_default_pdf or not args.pdf).lower(),
        "max_images": str(args.max_images),
        "max_new_tokens": str(args.max_new_tokens),
        "prompt_type": args.prompt_type,
        "use_answer_cache": str(args.use_answer_cache).lower(),
    }
    files = None
    handle = None
    if args.pdf:
        pdf_path = Path(args.pdf)
        handle = pdf_path.open("rb")
        files = {"file": (pdf_path.name, handle, "application/pdf")}
    start = time.perf_counter()
    try:
        response = requests.post(args.url, data=data, files=files, timeout=1800)
        elapsed_ms = (time.perf_counter() - start) * 1000
        ok = response.ok
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {"ok": ok, "status": response.status_code, "elapsed_ms": elapsed_ms, "payload": payload}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "status": "exception", "elapsed_ms": elapsed_ms, "error": str(exc)}
    finally:
        if handle is not None:
            handle.close()


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [item["elapsed_ms"] for item in results]
    ok_count = sum(1 for item in results if item["ok"])
    statuses: dict[str, int] = {}
    doc_hits = 0
    answer_hits = 0
    for item in results:
        statuses[str(item["status"])] = statuses.get(str(item["status"]), 0) + 1
        payload = item.get("payload") or {}
        doc_hits += int(bool(payload.get("document_cache_hit")))
        answer_hits += int(bool(payload.get("answer_cache_hit")))
    return {
        "requests": len(results),
        "success": ok_count,
        "success_rate": round(ok_count / max(1, len(results)), 4),
        "avg_ms": round(statistics.mean(latencies), 3) if latencies else 0,
        "p50_ms": round(percentile(latencies, 50), 3),
        "p90_ms": round(percentile(latencies, 90), 3),
        "p95_ms": round(percentile(latencies, 95), 3),
        "statuses": statuses,
        "document_cache_hits": doc_hits,
        "answer_cache_hits": answer_hits,
    }


def main() -> None:
    args = parse_args()
    levels = [int(item.strip()) for item in args.concurrency.split(",") if item.strip()]
    for level in levels:
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [pool.submit(call_once, args) for _ in range(args.requests_per_level)]
            results = [future.result() for future in as_completed(futures)]
        print({"concurrency": level, **summarize(results)})


if __name__ == "__main__":
    main()
