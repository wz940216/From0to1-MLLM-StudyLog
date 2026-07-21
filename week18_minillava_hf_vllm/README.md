# Week 18: MiniLLaVA HF Format and vLLM

This directory converts the `week16_multitask_tuning_summary` MiniLLaVA model into a Hugging Face style custom model directory.

## What Is Included

- `minillava_hf/configuration_minillava.py`: `MiniLlavaConfig`
- `minillava_hf/modeling_minillava.py`: `MiniLlavaForConditionalGeneration`
- `minillava_hf/processing_minillava.py`: `MiniLlavaProcessor`
- `scripts/convert_week16_to_hf.py`: exports week16 weights to `save_pretrained` format
- `scripts/infer_transformers.py`: local Transformers inference smoke test
- `scripts/vllm_openai_server.py`: reproducible vLLM OpenAI server launch command
- `scripts/vllm_chat_ui.py`: browser chat UI for the vLLM OpenAI server

The model keeps the week16 behavior: one `<image>` token in the prompt is replaced by all CLIP patch embeddings after a two-layer projector.

## Export From Week16

```bash
# convert trnsformers format
conda run -n mllm python week18_minillava_hf_vllm/scripts/convert_week16_to_hf.py \
  --config week16_multitask_tuning_summary/configs/multitask_balanced_dpo.yaml \
  --checkpoint week16_multitask_tuning_summary/outputs/checkpoints/multitask_balanced/dpo/step_1000.pt \
  --output-dir week18_minillava_hf_vllm/outputs/transformers/minillava-hf \
  --target "transformers"

# convert vllm format
conda run -n mllm python week18_minillava_hf_vllm/scripts/convert_week16_to_hf.py \
  --config week16_multitask_tuning_summary/configs/multitask_balanced_dpo.yaml \
  --checkpoint week16_multitask_tuning_summary/outputs/checkpoints/multitask_balanced/dpo/step_1000.pt \
  --output-dir week18_minillava_hf_vllm/outputs/vllm/minillava-hf \
  --target "vllm"

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
python week18_minillava_hf_vllm/scripts/infer_transformers.py --model-path week18_minillava_hf_vllm/outputs/transformers/minillava-hf --image dataset/coco128/images/train2017/000000000025.jpg --question "请描述这张图片。"

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
  --model-path week18_minillava_hf_vllm/outputs/vllm/minillava-hf \
  --served-model-name minillava \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85



# test
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minillava",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ],
    "max_tokens": 64
  }'


curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minillava",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片。"},
        {
          "type": "image_url",
          "uuid": "coco-000000000025",
          "image_url": {
            "url": "file:///gemini/data-1/code/mllm/dataset/coco128/images/train2017/000000000025.jpg"
          }
        }
      ]
    }],
    "max_tokens": 128
  }'

```


### Web Chat UI

Start the vLLM OpenAI server first, then start the local UI proxy:

```bash
python week18_minillava_hf_vllm/scripts/vllm_chat_ui.py \
  --vllm-base-url http://127.0.0.1:8000 \
  --model minillava \
  --host 127.0.0.1 \
  --port 7860
```
  
Open `http://127.0.0.1:7860` in a browser. The page supports one uploaded image per user turn and keeps the conversation history in the current browser session. Uploaded images are saved under `week18_minillava_hf_vllm/outputs/chat_uploads` and sent to vLLM as `file://` URLs, so keep the vLLM server's `--allowed-local-media-path` covering the project root or set `--upload-dir` to an allowed path.


## Notes

- Week16 LoRA checkpoints are loaded through the original week16 model first. If PEFT exposes `merge_and_unload`, the exporter merges the LoRA adapter into the base LLM before copying weights.
- The generated prompt must contain exactly one `<image>` token per sample.
- The current `generate` path delegates to the wrapped language model with `inputs_embeds`, which matches the original week16 implementation.
