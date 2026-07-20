import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


ROOT = Path(__file__).resolve().parents[2]
WEEK18_ROOT = ROOT / "week18_minillava_hf_vllm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WEEK18_ROOT) not in sys.path:
    sys.path.insert(0, str(WEEK18_ROOT))

from minillava_hf import register_minillava_auto_classes


def build_prompt(processor, question):
    return f"USER: {processor.image_token}\n{question}\nASSISTANT: "

def parse_args():
    parser = argparse.ArgumentParser(description="Run MiniLLaVA HF inference with Transformers.")
    parser.add_argument("--model-path", required=True, help="Path produced by convert_week16_to_hf.py.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--question", default="请描述这张图片。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False
    register_minillava_auto_classes()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(processor, args.question)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    print(processor.tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
