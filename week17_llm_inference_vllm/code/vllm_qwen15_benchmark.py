"""
    python week17_llm_inference_vllm/code/vllm_qwen15_benchmark.py bench-all \
        --num-requests 32 --concurrency 8 --max-new-tokens 64
"""

import argparse
import gc
import multiprocessing as mp
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "Qwen1.5-1.8B-Chat"
DEFAULT_PROMPT = "简单介绍一下大语言模型推理服务中连续批处理的作用。"
SYSTEM_PROMPT = "你是一个乐于助人的中文助手。"


def configure_cuda_multiprocessing() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)


@dataclass
class BenchResult:
    backend: str
    num_requests: int
    concurrency: int
    elapsed_sec: float
    qps: float
    output_tokens: int
    tokens_per_sec: float


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def print_result(result: BenchResult) -> None:
    print(f"\n[{result.backend}]")
    print(f"requests      : {result.num_requests}")
    print(f"concurrency   : {result.concurrency}")
    print(f"elapsed       : {result.elapsed_sec:.3f} s")
    print(f"qps           : {result.qps:.3f} req/s")
    print(f"output tokens : {result.output_tokens}")
    print(f"tokens/s      : {result.tokens_per_sec:.3f} tok/s")


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )


def resolve_torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype for Transformers: {dtype}")
    return dtype_map[dtype]


def count_new_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def benchmark_transformers(args: argparse.Namespace) -> BenchResult:
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(args.model_path)
    prompt = build_chat_prompt(tokenizer, args.prompt)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=resolve_torch_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    device = model.device

    def generate_once() -> str:
        inputs = {
            key: value.clone().to(device)
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    for _ in range(args.warmup):
        generate_once()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    output_tokens = 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(generate_once) for _ in range(args.num_requests)]
        for future in as_completed(futures):
            output_tokens += count_new_tokens(tokenizer, future.result())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return BenchResult(
        backend="transformers",
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        elapsed_sec=elapsed,
        qps=args.num_requests / elapsed,
        output_tokens=output_tokens,
        tokens_per_sec=output_tokens / elapsed,
    )


def benchmark_vllm(args: argparse.Namespace) -> BenchResult:
    configure_cuda_multiprocessing()

    from vllm import LLM, SamplingParams

    tokenizer = load_tokenizer(args.model_path)
    prompt = build_chat_prompt(tokenizer, args.prompt)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p,
    )
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.warmup > 0:
        llm.generate(
            [prompt] * args.warmup,
            sampling_params,
            use_tqdm=False,
        )

    prompts = [prompt] * args.num_requests
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start

    output_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
    return BenchResult(
        backend="vllm",
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        elapsed_sec=elapsed,
        qps=args.num_requests / elapsed,
        output_tokens=output_tokens,
        tokens_per_sec=output_tokens / elapsed,
    )


def run_vllm_server(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    print("Starting vLLM server:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def add_common_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="start a vLLM OpenAI-compatible API server")
    serve.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default="qwen1.5-1.8b-chat")
    serve.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    add_vllm_args(serve)

    bench_tf = subparsers.add_parser("bench-transformers", help="benchmark pure Transformers inference")
    add_common_bench_args(bench_tf)
    bench_tf.add_argument("--device-map", default="auto")

    bench_vllm = subparsers.add_parser("bench-vllm", help="benchmark offline vLLM inference")
    add_common_bench_args(bench_vllm)
    add_vllm_args(bench_vllm)

    bench_all = subparsers.add_parser("bench-all", help="run Transformers then vLLM benchmarks")
    add_common_bench_args(bench_all)
    bench_all.add_argument("--device-map", default="auto")
    add_vllm_args(bench_all)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command.startswith("bench") and args.num_requests < args.concurrency:
        raise ValueError("--num-requests should be >= --concurrency")
    if hasattr(args, "model_path") and not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")


def test_qps_compare(
    model_path: str = str(DEFAULT_MODEL_PATH),
    prompt: str = DEFAULT_PROMPT,
    num_requests: int = 32,
    concurrency: int = 8,
    max_new_tokens: int = 64,
    dtype: str = "bfloat16",
) -> tuple[BenchResult, BenchResult]:
    """Test single-prompt multi-concurrency QPS for Transformers and vLLM."""

    args = argparse.Namespace(
        command="bench-all",
        model_path=model_path,
        prompt=prompt,
        num_requests=num_requests,
        concurrency=concurrency,
        max_new_tokens=max_new_tokens,
        warmup=2,
        dtype=dtype,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        device_map="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
    )
    configure_cuda_multiprocessing()
    validate_args(args)
    transformers_result = benchmark_transformers(args)
    vllm_result = benchmark_vllm(args)
    print_result(transformers_result)
    print_result(vllm_result)
    print(f"\nvLLM / Transformers QPS speedup: {vllm_result.qps / transformers_result.qps:.2f}x")
    return transformers_result, vllm_result


def main(argv: Iterable[str] | None = None) -> None:
    configure_cuda_multiprocessing()

    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    if args.command == "serve":
        run_vllm_server(args)
    elif args.command == "bench-transformers":
        print_result(benchmark_transformers(args))
    elif args.command == "bench-vllm":
        print_result(benchmark_vllm(args))
    elif args.command == "bench-all":
        tf_result = benchmark_transformers(args)
        print_result(tf_result)
        vllm_result = benchmark_vllm(args)
        print_result(vllm_result)
        print(f"\nvLLM / Transformers QPS speedup: {vllm_result.qps / tf_result.qps:.2f}x")
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
