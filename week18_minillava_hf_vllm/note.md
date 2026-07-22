# week18_minillava_hf_vllm

第十六周训练的三阶段模型已经结束了，但是我们训练出来的模型参数仅保存了 project 以及 lora 的参数，image encoder 加载官方的权重参数。看起来很零散，也不容易部署和维护。

本周将对训练完毕的模型进行整理，将 lora 融合到 llm 里，将所有参数统一成 model.safetensors 的标准 Hugging Face 的风格，这样使用 transformer 的 AutoModelForCausalLM 即可快速加载模型，也容易通过 vllm 进行模型加载和启动服务。是模型上线或开源的最后一步。  

Week16 的 LoRA checkpoint 会先通过原始 week16 模型加载。如果 PEFT 暴露了 `merge_and_unload`，导出脚本会先将 LoRA adapter 合并到基础 LLM，再复制权重。生成使用的 prompt 中，每个样本必须且只能包含一个 `<image>` token。当前 `generate` 路径会用 `inputs_embeds` 委托给包装后的语言模型，保持与原始 week16 实现一致。  

导出的 hf 格式模型上传到网盘
```shell
通过网盘分享的文件：week18_minillava_hf_vllm
链接: https://pan.baidu.com/s/1nqhpABdB8S25ynQ-3OiIWQ?pwd=8888 提取码: 8888
```

## 目录内容

- `minillava_hf/configuration_minillava.py`：`MiniLlavaConfig`
- `minillava_hf/modeling_minillava.py`：`MiniLlavaForConditionalGeneration`
- `minillava_hf/processing_minillava.py`：`MiniLlavaProcessor`
- `scripts/convert_week16_to_hf.py`：将 week16 权重导出为 `save_pretrained` 格式
- `scripts/infer_transformers.py`：本地 Transformers 推理冒烟测试
- `scripts/vllm_openai_server.py`：可复现的 vLLM OpenAI 服务启动命令
- `scripts/vllm_chat_ui.py`：面向 vLLM OpenAI 服务的浏览器聊天界面

模型会保留 week16 的行为：提示词中的一个 `<image>` token 会在经过两层 projector 后，被替换为全部 CLIP patch embeddings。

在导出时，要实现一个 MiniLlavaConfig ，用于让 transformer 识别模型配置：

```python
from transformers import AutoConfig, CLIPVisionConfig, PretrainedConfig


class MiniLlavaConfig(PretrainedConfig):
    """Hugging Face style config for the week16 MiniLLaVA model."""

    model_type = "minillava"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projector_hidden_size=2048,
        image_token="<image>",
        image_token_id=None,
        ignore_index=-100,
        language_model_type="causal_lm",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            self.vision_config = CLIPVisionConfig()
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        else:
            self.vision_config = CLIPVisionConfig(**vision_config)

        if text_config is None:
            self.text_config = PretrainedConfig(hidden_size=2048, vocab_size=151936)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            model_type = text_config.get("model_type")
            if model_type:
                text_config = dict(text_config)
                text_config.pop("model_type", None)
                self.text_config = AutoConfig.for_model(model_type, **text_config)
            else:
                self.text_config = PretrainedConfig(**text_config)

        self.projector_hidden_size = projector_hidden_size
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.ignore_index = ignore_index
        self.language_model_type = language_model_type
        self.vocab_size = getattr(self.text_config, "vocab_size", None)
        self.hidden_size = getattr(self.text_config, "hidden_size", None)

    def to_dict(self):
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["language_model_type"] = self.language_model_type
        output["model_type"] = self.model_type
        return output


```

配置文件准备好之后，把模型定义构建好，这里的坑点在于 vllm 和 transformer 所需的模型字典不同有些地方需要调试和判断。

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, CLIPVisionModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_minillava import MiniLlavaConfig


class MiniLlavaProjector(nn.Module):
    """Two-layer MLP used by the week16 model to map CLIP patches to LLM hidden size."""

    def __init__(self, config: MiniLlavaConfig):
        super().__init__()
        vision_hidden_size = config.vision_config.hidden_size
        text_hidden_size = config.text_config.hidden_size
        self.fc1 = nn.Linear(vision_hidden_size, config.projector_hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.projector_hidden_size, text_hidden_size)
        self.norm = nn.LayerNorm(text_hidden_size)

    def forward(self, image_features):
        image_features = self.fc1(image_features)
        image_features = self.act(image_features)
        image_features = self.fc2(image_features)
        return self.norm(image_features)


class MiniLlavaPreTrainedModel(PreTrainedModel):
    config_class = MiniLlavaConfig
    base_model_prefix = "minillava"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CLIPEncoderLayer"]


class MiniLlavaForConditionalGeneration(MiniLlavaPreTrainedModel):
    """MiniLLaVA in a Transformers-compatible wrapper.

    Inputs follow the same rule as week16: each sample must contain exactly one
    ``<image>`` token. The token is replaced by all CLIP patch embeddings after
    projector mapping, then the resulting embeddings are passed to the CausalLM.
    """

    def __init__(self, config: MiniLlavaConfig):
        super().__init__(config)
        self.vision_tower = CLIPVisionModel(config.vision_config)
        self.multi_modal_projector = MiniLlavaProjector(config)
        self.language_model_type = getattr(config, "language_model_type", "causal_lm")
        if self.language_model_type == "backbone":
            self.language_model = AutoModel.from_config(
                config.text_config,
                trust_remote_code=True,
            )
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        else:
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config,
                trust_remote_code=True,
            )

    @property
    def all_tied_weights_keys(self):
        tied_keys = {}
        for attr_name in ("all_tied_weights_keys", "_tied_weights_keys"):
            keys = getattr(self.language_model, attr_name, None)
            if isinstance(keys, dict):
                keys = keys.keys()
            for key in keys or []:
                full_key = f"language_model.{key}"
                tied_keys[full_key] = full_key
        return tied_keys

    @classmethod
    def is_backend_compatible(cls):
        return True

    def get_image_features(self, pixel_values, **kwargs):
        return list(self._encode_images(pixel_values).unbind(0))

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        if hasattr(self, "lm_head"):
            return self.lm_head
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        if hasattr(self, "lm_head"):
            self.lm_head = new_embeddings
        else:
            self.language_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        try:
            embeddings = self.language_model.resize_token_embeddings(
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                mean_resizing=mean_resizing,
            )
        except TypeError:
            embeddings = self.language_model.resize_token_embeddings(
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
            )
        if hasattr(self, "lm_head") and new_num_tokens is not None:
            old_head = self.lm_head
            new_head = nn.Linear(
                old_head.in_features,
                int(new_num_tokens),
                bias=old_head.bias is not None,
                device=old_head.weight.device,
                dtype=old_head.weight.dtype,
            )
            rows = min(old_head.weight.size(0), new_head.weight.size(0))
            new_head.weight.data[:rows] = old_head.weight.data[:rows]
            if old_head.bias is not None:
                new_head.bias.data[:rows] = old_head.bias.data[:rows]
            self.lm_head = new_head
            self.config.text_config.vocab_size = int(new_num_tokens)
            self.config.vocab_size = int(new_num_tokens)
        return embeddings

    def _encode_images(self, pixel_values):
        vision_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        vision_outputs = self.vision_tower(pixel_values=pixel_values)
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]
        return self.multi_modal_projector(patch_features)

    def _merge_input_ids_with_image_features(
        self,
        image_features,
        input_ids,
        attention_mask,
        labels=None,
    ):
        if self.config.image_token_id is None:
            raise ValueError("config.image_token_id is not set. Save the tokenizer with the <image> token first.")

        text_embeddings = self.get_input_embeddings()(input_ids)
        image_token_id = self.config.image_token_id
        image_token_num = image_features.size(1)
        device = input_ids.device

        row_embeddings = []
        row_attention_masks = []
        row_labels = [] if labels is not None else None

        for row_idx in range(input_ids.size(0)):
            image_positions = (input_ids[row_idx] == image_token_id).nonzero(as_tuple=False).flatten()
            if image_positions.numel() != 1:
                raise ValueError("Each sample must contain exactly one <image> token.")

            image_pos = int(image_positions[0].item())
            row_embedding = torch.cat(
                [
                    text_embeddings[row_idx, :image_pos],
                    image_features[row_idx],
                    text_embeddings[row_idx, image_pos + 1:],
                ],
                dim=0,
            )
            row_attention_mask = torch.cat(
                [
                    attention_mask[row_idx, :image_pos],
                    torch.ones(image_token_num, dtype=attention_mask.dtype, device=device),
                    attention_mask[row_idx, image_pos + 1:],
                ],
                dim=0,
            )

            row_embeddings.append(row_embedding)
            row_attention_masks.append(row_attention_mask)

            if labels is not None:
                row_label = torch.cat(
                    [
                        labels[row_idx, :image_pos],
                        labels.new_full((image_token_num,), self.config.ignore_index),
                        labels[row_idx, image_pos + 1:],
                    ],
                    dim=0,
                )
                row_labels.append(row_label)

        max_length = max(row_embedding.size(0) for row_embedding in row_embeddings)
        hidden_size = row_embeddings[0].size(-1)
        inputs_embeds = image_features.new_zeros(len(row_embeddings), max_length, hidden_size)
        merged_attention_mask = attention_mask.new_zeros(len(row_embeddings), max_length)
        merged_labels = None
        if labels is not None:
            merged_labels = labels.new_full((len(row_embeddings), max_length), self.config.ignore_index)

        for row_idx, row_embedding in enumerate(row_embeddings):
            row_length = row_embedding.size(0)
            inputs_embeds[row_idx, :row_length] = row_embedding
            merged_attention_mask[row_idx, :row_length] = row_attention_masks[row_idx]
            if merged_labels is not None:
                merged_labels[row_idx, :row_length] = row_labels[row_idx]

        return inputs_embeds, merged_attention_mask, merged_labels

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            if input_ids is None or pixel_values is None:
                raise ValueError("MiniLLaVA forward requires input_ids and pixel_values.")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            image_features = self._encode_images(pixel_values)
            inputs_embeds, attention_mask, labels = self._merge_input_ids_with_image_features(
                image_features=image_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        language_dtype = next(self.language_model.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=language_dtype)
        # vLLM's Transformers backend wraps this class and expects backbone hidden
        # states, not LM logits. It passes attention_instances/return_dict=False.
        if labels is None and ("attention_instances" in kwargs or kwargs.get("return_dict") is False):
            base_model = getattr(self.language_model, "model", self.language_model)
            outputs = base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
            if isinstance(outputs, tuple):
                return outputs
            return (outputs.last_hidden_state,)

        if hasattr(self, "lm_head"):
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **generation_kwargs):
        if input_ids is None or pixel_values is None:
            raise ValueError("MiniLLaVA generate requires input_ids and pixel_values.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        image_features = self._encode_images(pixel_values)
        inputs_embeds, attention_mask, _ = self._merge_input_ids_with_image_features(
            image_features=image_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        language_dtype = next(self.language_model.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=language_dtype)
        if not hasattr(self.language_model, "generate"):
            raise RuntimeError("generate is only available for causal_lm exports; use vLLM for backbone exports.")
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
```

最后准备 processing：
```python
from typing import List, Optional, Union

from transformers import AutoTokenizer, CLIPImageProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

class MiniLlavaProcessor(ProcessorMixin):
    """Processor bundling the CLIP image processor and the LLM tokenizer."""

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, image_token="<image>", chat_template=None):
        self.image_token = image_token
        if tokenizer is not None:
            self._ensure_image_token(tokenizer, image_token)
            if chat_template is not None:
                tokenizer.chat_template = chat_template
            elif getattr(tokenizer, "chat_template", None) is None:
                tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template)

    @staticmethod
    def _ensure_image_token(tokenizer, image_token):
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        additional_special_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if image_token not in additional_special_tokens:
            tokenizer.add_special_tokens({
                "additional_special_tokens": additional_special_tokens + [image_token],
            })

    @property
    def image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        image_sizes = image_sizes or []
        image_processor = self.image_processor
        size = getattr(image_processor, "size", {}) or {}
        crop_size = getattr(image_processor, "crop_size", {}) or {}
        height = crop_size.get("height") or size.get("height") or size.get("shortest_edge") or 224
        width = crop_size.get("width") or size.get("width") or size.get("shortest_edge") or 224
        patch_size = 16
        num_patches = (int(height) // patch_size) * (int(width) // patch_size)
        count = len(image_sizes)
        return {
            "num_image_tokens": [num_patches] * count,
            "num_image_patches": [num_patches] * count,
        }

    @classmethod
    def from_components(cls, vision_model_path, tokenizer_model_path, image_token="<image>", **kwargs):
        image_processor = CLIPImageProcessor.from_pretrained(vision_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        kwargs.setdefault("chat_template", DEFAULT_CHAT_TEMPLATE)
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_token=image_token,
            **kwargs,
        )

    def __call__(
        self,
        text: Union[str, List[str], None] = None,
        images=None,
        return_tensors: Optional[str] = "pt",
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        encoding = {}
        return_mm_token_type_ids = bool(kwargs.pop("return_mm_token_type_ids", False))
        if text is not None:
            text_encoding = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs,
            )
            if return_mm_token_type_ids:
                input_rows = text_encoding["input_ids"].tolist()
                mask_rows = text_encoding.get("attention_mask")
                mask_rows = mask_rows.tolist() if mask_rows is not None else [[1] * len(row) for row in input_rows]
                image_count = len(images) if isinstance(images, list) else (1 if images is not None else 1)
                token_counts = self._get_num_multimodal_tokens(image_sizes=[(224, 224)] * image_count)["num_image_tokens"]
                expanded_input_rows = []
                expanded_mask_rows = []
                expanded_type_rows = []
                for row, mask in zip(input_rows, mask_rows):
                    out_ids = []
                    out_mask = []
                    out_types = []
                    replacement_idx = 0
                    for token_id, mask_value in zip(row, mask):
                        if token_id == self.image_token_id and replacement_idx < len(token_counts):
                            token_count = int(token_counts[replacement_idx])
                            out_ids.extend([self.image_token_id] * token_count)
                            out_mask.extend([mask_value] * token_count)
                            out_types.extend([1] * token_count)
                            replacement_idx += 1
                        else:
                            out_ids.append(token_id)
                            out_mask.append(mask_value)
                            out_types.append(0)
                    expanded_input_rows.append(out_ids)
                    expanded_mask_rows.append(out_mask)
                    expanded_type_rows.append(out_types)
                if return_tensors == "pt":
                    import torch
                    max_row_len = max(len(row) for row in expanded_input_rows)
                    pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
                    def pad(row, value):
                        return row + [value] * (max_row_len - len(row))
                    text_encoding["input_ids"] = torch.tensor([pad(row, pad_id) for row in expanded_input_rows], dtype=torch.long)
                    text_encoding["attention_mask"] = torch.tensor([pad(row, 0) for row in expanded_mask_rows], dtype=torch.long)
                    text_encoding["mm_token_type_ids"] = torch.tensor([pad(row, 0) for row in expanded_type_rows], dtype=torch.long)
                else:
                    text_encoding["input_ids"] = expanded_input_rows
                    text_encoding["attention_mask"] = expanded_mask_rows
                    text_encoding["mm_token_type_ids"] = expanded_type_rows
            encoding.update(text_encoding)
        if images is not None:
            image_encoding = self.image_processor(images=images, return_tensors=return_tensors)
            encoding.update(image_encoding)
        return BatchFeature(encoding)

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, **kwargs):
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


```
最后加载 MiniLLaVA 的 hf 模型，构建好 auto_map 后保存即可。


## 从 Week16 导出命令

```bash
# 转换为 Transformers 格式
conda run -n mllm python week18_minillava_hf_vllm/scripts/convert_week16_to_hf.py \
  --config week16_multitask_tuning_summary/configs/multitask_balanced_dpo.yaml \
  --checkpoint week16_multitask_tuning_summary/outputs/checkpoints/multitask_balanced/dpo/step_1000.pt \
  --output-dir week18_minillava_hf_vllm/outputs/transformers/minillava-hf \
  --target "transformers"

# 转换为 vLLM 格式
conda run -n mllm python week18_minillava_hf_vllm/scripts/convert_week16_to_hf.py \
  --config week16_multitask_tuning_summary/configs/multitask_balanced_dpo.yaml \
  --checkpoint week16_multitask_tuning_summary/outputs/checkpoints/multitask_balanced/dpo/step_1000.pt \
  --output-dir week18_minillava_hf_vllm/outputs/vllm/minillava-hf \
  --target "vllm"

```

`--checkpoint` 是可选参数。如果不提供该参数，导出脚本会根据 week16 配置保存基础 CLIP tower、基础 LLM，以及随机初始化的 projector。

导出脚本会保存：

- 通过 `save_pretrained` 保存的模型权重
- 已将 `<image>` 注册为额外特殊 token 的 tokenizer
- CLIP image processor 文件
- 自定义模型代码和 `auto_map`
- `conversion_meta.json`

## Transformers 推理

```bash
python week18_minillava_hf_vllm/scripts/infer_transformers.py --model-path week18_minillava_hf_vllm/outputs/transformers/minillava-hf --image dataset/coco128/images/train2017/000000000025.jpg --question "请描述这张图片。"

A giraffe and a tree by the side of a trail.
```

注册本地类后，也可以通过 Auto classes 加载：

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

# 测试
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
            "url": "file:///path/to/dataset/coco128/images/train2017/000000000025.jpg"
          }
        }
      ]
    }],
    "max_tokens": 128
  }'

```

### Web 聊天界面
先启动 vLLM OpenAI 服务，然后启动本地 UI 代理：
```bash
python week18_minillava_hf_vllm/scripts/vllm_chat_ui.py \
  --vllm-base-url http://127.0.0.1:8000 \
  --model minillava \
  --host 127.0.0.1 \
  --port 7860
```

在浏览器中打开 `http://127.0.0.1:7860`。页面支持每轮用户消息上传一张图片，并会在当前浏览器会话中保留对话历史。上传的图片会保存到 `week18_minillava_hf_vllm/outputs/chat_uploads`，并以 `file://` URL 的形式发送给 vLLM。因此，需要确保 vLLM 服务的 `--allowed-local-media-path` 覆盖项目根目录，或者将 `--upload-dir` 设置为 vLLM 允许访问的路径。

麻雀虽小现在五脏俱全，测试通过后我们 minillava 就可以开源和发版了。标准的 hf 格式，可通过 vllm 部署。
