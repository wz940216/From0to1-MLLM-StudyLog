import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoConfig, CLIPVisionConfig


ROOT = Path(__file__).resolve().parents[2]
WEEK18_ROOT = ROOT / "week18_minillava_hf_vllm"
WEEK16_CODE = ROOT / "week16_multitask_tuning_summary" / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WEEK18_ROOT) not in sys.path:
    sys.path.insert(0, str(WEEK18_ROOT))
if str(WEEK16_CODE) not in sys.path:
    sys.path.insert(0, str(WEEK16_CODE))

from minillava_hf import MiniLlavaConfig, MiniLlavaForConditionalGeneration, MiniLlavaProcessor
from week16_multitask_tuning_summary.code.mini_llava import MiniLlavaModel as Week16MiniLlavaModel


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_week16_checkpoint(model, checkpoint_path):
    if checkpoint_path is None:
        return
    state = torch.load(checkpoint_path, map_location=model.device)
    state_dict = state.get("model", state)
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Missing keys: {len(incompatible.missing_keys)}")
    print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")


def unwrap_language_model(language_decoder):
    model = language_decoder.model
    if hasattr(model, "merge_and_unload"):
        print("Merging PEFT LoRA adapter into the base language model.")
        model = model.merge_and_unload()
    return model


def copy_module_state(target, source, name):
    incompatible = target.load_state_dict(source.state_dict(), strict=False)
    print(f"[{name}] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")


def build_hf_config(config, processor, target="transformers"):
    vision_path = config["MINILLAVA"]["VISION_ENCODER"]["MODEL_PATH"]
    llm_path = config["MINILLAVA"]["LLM_DECODER"]["MODEL_PATH"]
    vision_config = CLIPVisionConfig.from_pretrained(vision_path)
    text_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
    return MiniLlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        projector_hidden_size=int(config["MINILLAVA"]["PROJECTOR"]["HIDDEN_DIM"]),
        image_token=processor.image_token,
        image_token_id=processor.tokenizer.convert_tokens_to_ids(processor.image_token),
        language_model_type="backbone" if target == "vllm" else "causal_lm",
        pad_token_id=processor.tokenizer.pad_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )


def copy_custom_code(output_dir):
    package_dir = ROOT / "week18_minillava_hf_vllm" / "minillava_hf"
    for file_name in [
        "__init__.py",
        "configuration_minillava.py",
        "modeling_minillava.py",
        "processing_minillava.py",
    ]:
        shutil.copy2(package_dir / file_name, output_dir / file_name)


def normalize_vllm_export_files(output_dir):
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        if isinstance(tokenizer_config.get("extra_special_tokens"), list):
            tokenizer_config["extra_special_tokens"] = {"image_token": "<image>"}
        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
            f.write("\n")

    processor_config_path = output_dir / "processor_config.json"
    if processor_config_path.exists():
        with open(processor_config_path, "r", encoding="utf-8") as f:
            processor_config = json.load(f)
        image_processor_config = processor_config.pop("image_processor", None)
        processor_config["auto_map"] = {"AutoProcessor": "processing_minillava.MiniLlavaProcessor"}
        with open(processor_config_path, "w", encoding="utf-8") as f:
            json.dump(processor_config, f, ensure_ascii=False, indent=2)
            f.write("\n")
        if image_processor_config is not None:
            image_processor_config.setdefault("image_processor_type", "CLIPImageProcessor")
            with open(output_dir / "preprocessor_config.json", "w", encoding="utf-8") as f:
                json.dump(image_processor_config, f, ensure_ascii=False, indent=2)
                f.write("\n")


def rewrite_vllm_safetensors_keys(output_dir):
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ImportError("--target vllm requires safetensors to rewrite weight keys.") from exc

    model_path = output_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected single-shard safetensors file: {model_path}")

    tensors = load_file(str(model_path))
    rewritten = {}
    for name, tensor in tensors.items():
        if name.startswith(("language_model.model.", "language_model.lm_head.")):
            new_name = name
        elif name.startswith("language_model."):
            new_name = "language_model.model." + name[len("language_model."):]
        elif name.startswith("lm_head."):
            new_name = "language_model.lm_head." + name[len("lm_head."):]
        elif name.startswith("vision_tower.") and not name.startswith("vision_tower.vision_model."):
            new_name = "vision_tower.vision_model." + name[len("vision_tower."):]
        else:
            new_name = name
        rewritten[new_name] = tensor

    lm_head_tensor = rewritten.get("language_model.lm_head.weight")
    if lm_head_tensor is not None:
        rewritten.setdefault("model.lm_head.weight", lm_head_tensor.clone())

    save_file(rewritten, str(model_path))


def write_readme(output_dir, source_config, checkpoint_path, target="transformers"):
    metadata = {
        "source_config": str(source_config),
        "source_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "format": "MiniLLaVA HF custom code",
        "target": target,
    }
    with open(output_dir / "conversion_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(
            "# MiniLLaVA HF Export\n\n"
            "This directory was exported from `week16_multitask_tuning_summary` by "
            "`week18_minillava_hf_vllm/scripts/convert_week16_to_hf.py`.\n\n"
            "Load with `trust_remote_code=True`, or import `minillava_hf` and call "
            "`register_minillava_auto_classes()` before using Auto classes.\n"
        )


def convert(args):
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = MiniLlavaProcessor.from_components(
        vision_model_path=config["MINILLAVA"]["VISION_ENCODER"]["MODEL_PATH"],
        tokenizer_model_path=config["MINILLAVA"]["LLM_DECODER"]["MODEL_PATH"],
    )
    hf_config = build_hf_config(config, processor, target=args.target)

    print("Loading week16 model.")
    week16_model = Week16MiniLlavaModel(str(args.config))
    load_week16_checkpoint(week16_model, args.checkpoint)
    week16_model.eval()

    print("Building HF model.")
    hf_model = MiniLlavaForConditionalGeneration(hf_config)
    hf_model.resize_token_embeddings(len(processor.tokenizer))

    copy_module_state(hf_model.vision_tower, week16_model.vision_encoder.vision_model, "vision_tower")
    copy_module_state(hf_model.multi_modal_projector, week16_model.projector, "multi_modal_projector")
    source_language_model = unwrap_language_model(week16_model.language_decoder)
    if args.target == "vllm":
        copy_module_state(hf_model.language_model, source_language_model.model, "language_model_backbone")
        copy_module_state(hf_model.lm_head, source_language_model.lm_head, "lm_head")
    else:
        copy_module_state(hf_model.language_model, source_language_model, "language_model")

    hf_model.config.auto_map = {
        "AutoConfig": "configuration_minillava.MiniLlavaConfig",
        "AutoModel": "modeling_minillava.MiniLlavaForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_minillava.MiniLlavaForConditionalGeneration",
        "AutoProcessor": "processing_minillava.MiniLlavaProcessor",
    }
    hf_model.config.architectures = ["MiniLlavaForConditionalGeneration"]
    hf_model.save_pretrained(output_dir, safe_serialization=True)
    if args.target == "vllm":
        rewrite_vllm_safetensors_keys(output_dir)
    processor.save_pretrained(output_dir)
    normalize_vllm_export_files(output_dir)
    copy_custom_code(output_dir)
    write_readme(output_dir, args.config, args.checkpoint, target=args.target)
    print(f"Saved HF MiniLLaVA model to: {output_dir}")
    if args.target != "vllm":
        print("Note: vLLM serving requires a separate export with --target vllm.")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert week16 MiniLLaVA checkpoint to HF custom model format.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "week16_multitask_tuning_summary" / "configs" / "config.yaml",
        help="week16 YAML config path.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional week16 .pt checkpoint.")
    parser.add_argument("--target", choices=["transformers", "vllm"], default="transformers", help="Export layout target. Use vllm for vLLM Transformers backend.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "week18_minillava_hf_vllm" / "outputs" / "minillava-hf",
        help="HF model output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
