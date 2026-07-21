import torch
from vision_encoder import VisionEncoder
from llm_decoder import LLMDecoder, Projector
import yaml


class MiniLlavaModel(torch.nn.Module):
    """MiniLLaVA

    模型由三部分组成：
    1. CLIP Vision Encoder：把图片编码成 patch 特征。
    2. Projector：把视觉特征映射到 LLM hidden_size。
    3. LLM Decoder：把图片 embedding + 文本 embedding 当作上下文，学习生成回答。
    """

    def __init__(self, config_path):
        super(MiniLlavaModel, self).__init__()
        self.config = self.load_config(config_path)
        self.device = self._resolve_device(self.config.get("DEVICE", "cuda"))

        self.vision_encoder = VisionEncoder(
            model_path=self.config["MINILLAVA"]["VISION_ENCODER"]["MODEL_PATH"],
            freeze=self.config["MINILLAVA"]["VISION_ENCODER"].get("FREEZE", True),
            device=str(self.device)
        )
        self.language_decoder = LLMDecoder(
            r=self.config["MINILLAVA"]["LLM_DECODER"].get("LORA_R", 8),
            lora_alpha=self.config["MINILLAVA"]["LLM_DECODER"].get("LORA_ALPHA", 32),
            lora_dropout=self.config["MINILLAVA"]["LLM_DECODER"].get("LORA_DROPOUT", 0.1),
            model_path=self.config["MINILLAVA"]["LLM_DECODER"]["MODEL_PATH"],
            freeze=self.config["MINILLAVA"]["LLM_DECODER"].get("FREEZE", False),
            device=str(self.device)
        )
        self.projector = Projector(
            input_dim=self.config["MINILLAVA"]["PROJECTOR"]["INPUT_DIM"],
            hidden_dim=self.config["MINILLAVA"]["PROJECTOR"]["HIDDEN_DIM"],
            output_dim=self.language_decoder.model.config.hidden_size,
            freeze=self.config["MINILLAVA"]["PROJECTOR"].get("FREEZE", False)
        ).to(self.device)

    def _resolve_device(self, device):
        """把配置中的设备字符串转成真实可用的 torch.device。

        accelerate 多进程启动时，每个进程都有自己的 LOCAL_RANK。这里不能只返回
        cuda，否则模型在 accelerator.prepare() 之前会先被搬到 cuda:0，4 卡训练时
        容易在同一张卡上初始化 cuDNN/显存。
        """
        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("配置使用 cuda，但当前环境没有可用 GPU，自动切换到 cpu。")
            return torch.device("cpu")
        if device == "cuda" and torch.cuda.is_available():
            import os

            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                local_rank = int(local_rank)
                torch.cuda.set_device(local_rank)
                return torch.device(f"cuda:{local_rank}")
        return torch.device(device)

    def load_config(self, config_path):
        """读取 YAML 配置文件。"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def _build_multimodal_inputs(self, images, input_ids, attention_mask, labels=None):
        """在 <image> token 位置展开视觉 patch embedding。"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # 1. input_ids -> 文本 embedding。这里复用 LLM 自己的词向量表，保证空间一致。
        text_embeddings = self.language_decoder.get_input_embeddings()(input_ids)

        # 2. 图片 -> CLIP patch 特征 -> projector -> LLM hidden_size。
        # generate() 会直接把 inputs_embeds 传给底层 LLM，必须提前对齐到 LLM embedding dtype。
        projector_dtype = next(self.projector.parameters()).dtype
        image_features = self.vision_encoder(images).to(dtype=projector_dtype)
        projected_image_features = self.projector(image_features).to(dtype=text_embeddings.dtype)
        image_token_id = self.language_decoder.image_token_id
        image_token_num = projected_image_features.size(1)

        row_embeddings = []
        row_attention_masks = []
        row_labels = [] if labels is not None else None
        
        # batch 内每条样本单独处理，找到 <image> token 位置，在那里插入视觉特征，并调整 attention mask 和 labels。
        for row_idx in range(input_ids.size(0)):
            image_positions = (input_ids[row_idx] == image_token_id).nonzero(as_tuple=False).flatten()
            if image_positions.numel() != 1:
                raise ValueError("每条样本必须包含且只包含一个 <image> token。")

            image_pos = int(image_positions[0].item())
            row_image_features = projected_image_features[row_idx]
            row_embedding = torch.cat(
                [
                    text_embeddings[row_idx, :image_pos],
                    row_image_features,
                    text_embeddings[row_idx, image_pos + 1:]
                ],
                dim=0
            )
            image_attention_mask = torch.ones(
                image_token_num,
                dtype=attention_mask.dtype,
                device=self.device
            )
            row_attention_mask = torch.cat(
                [
                    attention_mask[row_idx, :image_pos],
                    image_attention_mask,
                    attention_mask[row_idx, image_pos + 1:]
                ],
                dim=0
            )

            row_embeddings.append(row_embedding)
            row_attention_masks.append(row_attention_mask)

            if labels is not None:
                image_labels = torch.full(
                    (image_token_num,),
                    -100,
                    dtype=torch.long,
                    device=self.device
                )
                row_label = torch.cat(
                    [
                        labels[row_idx, :image_pos],
                        image_labels,
                        labels[row_idx, image_pos + 1:]
                    ],
                    dim=0
                )
                row_labels.append(row_label)

        max_length = max(row_embedding.size(0) for row_embedding in row_embeddings)
        hidden_size = row_embeddings[0].size(-1)
        
        # batch 内不同样本的长度可能不一样，这里统一 pad 到 max_length。视觉特征部分的 padding 不会被 attention mask 和 labels 关注到。
        combined_inputs = text_embeddings.new_zeros(
            len(row_embeddings),
            max_length,
            hidden_size
        )
        combined_attention_mask = attention_mask.new_zeros(len(row_embeddings), max_length)
        combined_labels = None
        if labels is not None:
            combined_labels = labels.new_full((len(row_embeddings), max_length), -100)

        for row_idx, row_embedding in enumerate(row_embeddings):
            row_length = row_embedding.size(0)
            combined_inputs[row_idx, :row_length] = row_embedding
            combined_attention_mask[row_idx, :row_length] = row_attention_masks[row_idx]
            if combined_labels is not None:
                combined_labels[row_idx, :row_length] = row_labels[row_idx]

        return combined_inputs, combined_attention_mask, combined_labels

    def forward(self, images, input_ids=None, attention_mask=None, labels=None, texts=None):
        """训练/调试共用 forward。

        推荐训练时直接传入 dataset 已经构造好的 input_ids、attention_mask、labels；
        为了兼容原来的调试方式，也保留 texts 参数。
        """
        if texts is not None:
            # 旧接口：把整段文本都作为监督目标。正式微调请使用 dataset.py 生成 labels，
            # 因为它会屏蔽用户问题，只监督助手回答。
            image_token = self.language_decoder.image_token
            texts = [
                text if image_token in text else f"USER: {image_token}\n{text}\nASSISTANT: "
                for text in texts
            ]
            tokenized = self.language_decoder.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["DATA"]["PREPROCESS"]["MAX_TEXT_LENGTH"]
            )
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            labels = input_ids.clone()

        if input_ids is None or attention_mask is None:
            raise ValueError("必须传入 input_ids/attention_mask，或传入 texts 走调试路径。")

        inputs_embeds, combined_attention_mask, combined_labels = self._build_multimodal_inputs(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        outputs = self.language_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            use_cache=False # 训练时不启用kv catche
        )
        return outputs

    @torch.no_grad()
    def generate(self, images, prompts, max_new_tokens=128, **generation_kwargs):
        """多模态生成接口，用于训练后的简单验证。"""
        self.eval()
        image_token = self.language_decoder.image_token
        prompts = [
            prompt if image_token in prompt else f"USER: {image_token}\n{prompt}\nASSISTANT: "
            for prompt in prompts
        ]
        tokenized = self.language_decoder.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config["DATA"]["PREPROCESS"]["MAX_TEXT_LENGTH"]
        )
        inputs_embeds, attention_mask, _ = self._build_multimodal_inputs(
            images=images,
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask
        )
        output_ids = self.language_decoder.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.language_decoder.tokenizer.pad_token_id,
            eos_token_id=self.language_decoder.tokenizer.eos_token_id,
            **generation_kwargs
        )
        return self.language_decoder.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


if __name__ == "__main__":
    model = MiniLlavaModel(config_path="week07_minillava_design_data/code/config.yaml")
    from PIL import Image
    image = Image.open("dataset/coco128/images/train2017/000000000009.jpg").convert("RGB")
    dummy_images = [image,image,image]  # Simulating a batch of 3 images
    dummy_texts = ["What is in the image?"] * 3  # Simulating a batch of 3 identical questions
    outputs = model(images=dummy_images, texts=dummy_texts)
    
    print(outputs.logits.shape) #logits.shape = (batch_size, seq_len, vocab_size) vocab_size 是词表大小
    # torch.Size([3, 202, 151936])
    # 训练时 use_cache=False，因此 outputs.past_key_values 默认为 None；
    # 推理生成时 generate 会自动使用 KV cache 加速。
    
    print(outputs.loss)
    # tensor(9.4443, device='cuda:0', grad_fn=<NllLossBackward0>)
    
    
