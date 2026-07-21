"""Start an OpenAI-compatible vLLM server for the exported MiniLLaVA directory."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "week18_minillava_hf_vllm" / "outputs" / "minillava-hf"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI API server for MiniLLaVA HF export.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--served-model-name", default="minillava")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--model-impl", default="transformers", choices=["auto", "vllm", "transformers"])
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--allowed-local-media-path", default=str(ROOT))
    parser.add_argument("--chat-template-content-format", default="openai", choices=["auto", "string", "openai"])
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = str(Path(args.model_path).resolve())
    conversion_meta_path = Path(model_path) / "conversion_meta.json"
    if conversion_meta_path.exists():
        with open(conversion_meta_path, "r", encoding="utf-8") as f:
            conversion_meta = json.load(f)
        if conversion_meta.get("target") != "vllm":
            raise SystemExit(
                f"{model_path} was exported with target="
                f"{conversion_meta.get('target')!r}. Re-run "
                "convert_week16_to_hf.py with --target vllm before starting vLLM."
            )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--tokenizer",
        model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--model-impl",
        args.model_impl,
        "--max-model-len",
        str(args.max_model_len),
        "--allowed-local-media-path",
        args.allowed_local_media_path,
        "--chat-template-content-format",
        args.chat_template_content_format,
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
