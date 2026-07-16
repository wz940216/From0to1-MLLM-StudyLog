"""
Start an OpenAI-compatible vLLM server for the exported MiniLLaVA directory.

vLLM must support this custom architecture before this can run end to end. The
script keeps the launch command reproducible for week18; if your vLLM version
does not recognize `minillava`, use the Transformers inference script first.
"""

import argparse
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
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
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
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
