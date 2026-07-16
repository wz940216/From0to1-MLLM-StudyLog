import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, CLIPVisionModel, PreTrainedModel
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
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            trust_remote_code=True,
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        try:
            return self.language_model.resize_token_embeddings(
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
                mean_resizing=mean_resizing,
            )
        except TypeError:
            return self.language_model.resize_token_embeddings(
                new_num_tokens=new_num_tokens,
                pad_to_multiple_of=pad_to_multiple_of,
            )

    def _encode_images(self, pixel_values):
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
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
