from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor

from .configuration_minillava import MiniLlavaConfig
from .modeling_minillava import MiniLlavaForConditionalGeneration
from .processing_minillava import MiniLlavaProcessor


def register_minillava_auto_classes():
    AutoConfig.register(MiniLlavaConfig.model_type, MiniLlavaConfig)
    AutoModel.register(MiniLlavaConfig, MiniLlavaForConditionalGeneration)
    AutoModelForCausalLM.register(MiniLlavaConfig, MiniLlavaForConditionalGeneration)
    AutoProcessor.register(MiniLlavaConfig, MiniLlavaProcessor)


__all__ = [
    "MiniLlavaConfig",
    "MiniLlavaForConditionalGeneration",
    "MiniLlavaProcessor",
    "register_minillava_auto_classes",
]
