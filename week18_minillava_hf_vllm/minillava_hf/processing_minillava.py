from typing import List, Optional, Union

from transformers import AutoTokenizer, CLIPImageProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}SYSTEM: {% if message['content'] is string %}{{ message['content'] }}{% else %}{% for block in message['content'] %}{% if block['type'] == 'text' %}{{ block['text'] }}{% endif %}{% endfor %}{% endif %}\n{% elif message['role'] == 'user' %}USER: {% if message['content'] is string %}{{ message['content'] }}{% else %}{% for block in message['content'] %}{% if block['type'] == 'image' or block['type'] == 'image_url' %}<image>\n{% elif block['type'] == 'text' %}{{ block['text'] }}{% endif %}{% endfor %}{% endif %}\n{% elif message['role'] == 'assistant' %}ASSISTANT: {% if message['content'] is string %}{{ message['content'] }}{% else %}{% for block in message['content'] %}{% if block['type'] == 'text' %}{{ block['text'] }}{% endif %}{% endfor %}{% endif %}\n{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"


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
