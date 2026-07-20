# Week 18: MiniLLaVA HF Format and vLLM

This directory converts the `week16_multitask_tuning_summary` MiniLLaVA model into a Hugging Face style custom model directory.

## What Is Included

- `minillava_hf/configuration_minillava.py`: `MiniLlavaConfig`
- `minillava_hf/modeling_minillava.py`: `MiniLlavaForConditionalGeneration`
- `minillava_hf/processing_minillava.py`: `MiniLlavaProcessor`
- `scripts/convert_week16_to_hf.py`: exports week16 weights to `save_pretrained` format
- `scripts/infer_transformers.py`: local Transformers inference smoke test
- `scripts/vllm_openai_server.py`: reproducible vLLM OpenAI server launch command

The model keeps the week16 behavior: one `<image>` token in the prompt is replaced by all CLIP patch embeddings after a two-layer projector.

## Export From Week16

```bash
python week18_minillava_hf_vllm/scripts/convert_week16_to_hf.py \
  --config week16_multitask_tuning_summary/configs/multitask_balanced_sft.yaml \
  --checkpoint week16_multitask_tuning_summary/outputs/checkpoints/multitask_balanced/sft/best.pt \
  --output-dir week18_minillava_hf_vllm/outputs/minillava-hf
```

`--checkpoint` is optional. Without it, the exporter saves the base CLIP tower, base LLM, and randomly initialized projector from the week16 config.

The exporter saves:

- model weights via `save_pretrained`
- tokenizer with `<image>` registered as an additional special token
- CLIP image processor files
- custom model code and `auto_map`
- `conversion_meta.json`

## Transformers Inference

```bash
python week18_minillava_hf_vllm/scripts/infer_transformers.py   --model-path week18_minillava_hf_vllm/outputs/minillava-hf   --image dataset/coco128/images/train2017/000000000025.jpg   --question 
"请描述这张图片。"

A giraffe and a tree by the side of a trail.
```

You can also load it through Auto classes after registering the local classes:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from week18_minillava_hf_vllm.minillava_hf import register_minillava_auto_classes

register_minillava_auto_classes()
processor = AutoProcessor.from_pretrained("week18_minillava_hf_vllm/outputs/minillava-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("week18_minillava_hf_vllm/outputs/minillava-hf", trust_remote_code=True)
```

## vLLM

```bash
conda run -n vllm_test python week18_minillava_hf_vllm/scripts/vllm_openai_server.py \
  --model-path week18_minillava_hf_vllm/outputs/minillava-hf \
  --served-model-name minillava \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.3
```


## Notes

- Week16 LoRA checkpoints are loaded through the original week16 model first. If PEFT exposes `merge_and_unload`, the exporter merges the LoRA adapter into the base LLM before copying weights.
- The generated prompt must contain exactly one `<image>` token per sample.
- The current `generate` path delegates to the wrapped language model with `inputs_embeds`, which matches the original week16 implementation.
