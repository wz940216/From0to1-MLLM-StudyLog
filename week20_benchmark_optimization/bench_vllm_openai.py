#!/usr/bin/env python3
"""Direct OpenAI-compatible benchmark for MiniLLaVA vLLM service.

This avoids version-specific `vllm bench serve` dataset support differences and
sends the same multimodal Chat Completions shape used by the week19 service.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM OpenAI-compatible multimodal chat endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="minillava")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="请描述这张图片。")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rate", type=float, default=2.0, help="Requests per second. Use <=0 for no pacing.")
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--result-dir", default="week20_benchmark_optimization/results/vllm")
    parser.add_argument("--result-filename", default="")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def load_image_data_url(image_path: Path) -> tuple[str, str]:
    data = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    image_uuid = hashlib.sha256(data).hexdigest()
    return f"data:{mime};base64,{encoded}", image_uuid


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> tuple[int, dict[str, Any] | None, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            text = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(text), text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, text
    except Exception as exc:  # noqa: BLE001 - benchmark should capture failures as results.
        return 0, None, repr(exc)


def run_one(index: int, args: argparse.Namespace, image_data_url: str, image_uuid: str) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "uuid": image_uuid, "image_url": {"url": image_data_url}},
                    {"type": "text", "text": args.question},
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    started = time.perf_counter()
    status, data, raw_text = post_json(url, payload, args.timeout_s)
    latency_ms = (time.perf_counter() - started) * 1000.0
    ok = status == 200 and isinstance(data, dict) and bool(data.get("choices"))
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    answer = ""
    if ok:
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            ok = False
    return {
        "index": index,
        "ok": ok,
        "status": status,
        "latency_ms": round(latency_ms, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "answer_chars": len(answer) if answer else 0,
        "error": "" if ok else raw_text[:1000],
    }


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"image does not exist: {image_path}")

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / (args.result_filename or f"openai_vllm_{args.model}_{int(time.time())}.jsonl")
    summary_file = result_file.with_suffix(".summary.json")

    image_data_url, image_uuid = load_image_data_url(image_path)
    interval_s = 0.0 if args.request_rate <= 0 else 1.0 / args.request_rate
    next_submit_ts = time.perf_counter()
    start_ts = time.perf_counter()
    results: list[dict[str, Any]] = []

    print("Direct vLLM OpenAI-compatible benchmark")
    print(f"  base_url        {args.base_url}")
    print(f"  model           {args.model}")
    print(f"  image           {image_path}")
    print(f"  num_prompts     {args.num_prompts}")
    print(f"  request_rate    {args.request_rate}")
    print(f"  max_concurrency {args.max_concurrency}")
    print(f"  max_tokens      {args.max_tokens}")
    print(f"  result_file     {result_file}")

    with result_file.open("w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            futures: list[concurrent.futures.Future[dict[str, Any]]] = []
            for index in range(args.num_prompts):
                if interval_s > 0:
                    sleep_s = next_submit_ts - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    next_submit_ts += interval_s
                futures.append(executor.submit(run_one, index, args, image_data_url, image_uuid))

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                if not result["ok"]:
                    print(f"FAILED index={result['index']} status={result['status']} error={result['error'][:200]}")

    total_s = time.perf_counter() - start_ts
    ok_results = [r for r in results if r["ok"]]
    latencies = [float(r["latency_ms"]) for r in ok_results]
    completion_tokens = [r.get("completion_tokens") or 0 for r in ok_results]
    summary = {
        "total_requests": len(results),
        "successful_requests": len(ok_results),
        "failed_requests": len(results) - len(ok_results),
        "total_time_s": round(total_s, 3),
        "request_throughput": round(len(ok_results) / total_s, 6) if total_s > 0 else 0,
        "output_token_throughput": round(sum(completion_tokens) / total_s, 6) if total_s > 0 else 0,
        "latency_ms_avg": round(statistics.mean(latencies), 3) if latencies else 0,
        "latency_ms_p50": round(percentile(latencies, 50), 3),
        "latency_ms_p90": round(percentile(latencies, 90), 3),
        "latency_ms_p95": round(percentile(latencies, 95), 3),
        "latency_ms_p99": round(percentile(latencies, 99), 3),
        "result_file": str(result_file),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
