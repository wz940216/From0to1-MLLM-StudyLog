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
        self.vocab_size = getattr(self.text_config, "vocab_size", None)
        self.hidden_size = getattr(self.text_config, "hidden_size", None)

    def to_dict(self):
        output = super().to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.model_type
        return output
