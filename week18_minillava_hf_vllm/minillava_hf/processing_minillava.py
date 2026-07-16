from typing import List, Optional, Union

from transformers import AutoTokenizer, CLIPImageProcessor, ProcessorMixin


DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}SYSTEM: {{ message['content'] }}
{% elif message['role'] == 'user' %}USER: {% if message['content'] is string %}{{ message['content'] }}{% else %}{% for block in message['content'] %}{% if block['type'] == 'image' %}<image>
{% elif block['type'] == 'text' %}{{ block['text'] }}{% endif %}{% endfor %}{% endif %}
{% elif message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""


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
        additional_special_tokens = list(tokenizer.additional_special_tokens)
        if image_token not in additional_special_tokens:
            tokenizer.add_special_tokens({
                "additional_special_tokens": additional_special_tokens + [image_token],
            })

    @classmethod
    def from_components(cls, vision_model_path, tokenizer_model_path, image_token="<image>", **kwargs):
        image_processor = CLIPImageProcessor.from_pretrained(vision_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_path,
            trust_remote_code=True,
            use_fast=True,
        )
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
        if text is not None:
            text_encoding = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs,
            )
            encoding.update(text_encoding)
        if images is not None:
            image_encoding = self.image_processor(images=images, return_tensors=return_tensors)
            encoding.update(image_encoding)
        return encoding

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True, **kwargs):
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
