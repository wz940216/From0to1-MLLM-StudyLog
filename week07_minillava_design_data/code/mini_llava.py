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
            model_path=self.config["MINILLAVA"]["LLM_DECODER"]["MODEL_PATH"],
            freeze=self.config["MINILLAVA"]["LLM_DECODER"].get("FREEZE", False),
            device=str(self.device)
        )
        self.projector = Projector(
            input_dim=self.config["MINILLAVA"]["PROJECTOR"]["INPUT_DIM"],
            hidden_dim=self.config["MINILLAVA"]["PROJECTOR"]["HIDDEN_DIM"],
            output_dim=self.language_decoder.model.config.hidden_size
        ).to(self.device)

    def _resolve_device(self, device):
        """把配置中的设备字符串转成真实可用的 torch.device。"""
        if device == "cuda" and not torch.cuda.is_available():
            print("配置使用 cuda，但当前环境没有可用 GPU，自动切换到 cpu。")
            return torch.device("cpu")
        return torch.device(device)

    def load_config(self, config_path):
        """读取 YAML 配置文件。"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def _build_multimodal_inputs(self, images, input_ids, attention_mask):
        """构造 LLM 可直接接收的多模态 embedding 和 attention mask。"""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # 1. 图片 -> CLIP patch 特征 -> projector -> LLM hidden_size。
        image_features = self.vision_encoder(images)
        projected_image_features = self.projector(image_features)

        # 2. input_ids -> 文本 embedding。这里复用 LLM 自己的词向量表，保证空间一致。
        text_embeddings = self.language_decoder.get_input_embeddings()(input_ids)

        # 3. 图片 patch 没有 padding，attention mask 全部置 1。
        image_attention_mask = torch.ones(
            projected_image_features.size(0),
            projected_image_features.size(1),
            dtype=attention_mask.dtype,
            device=self.device
        )
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        # 4. 序列维度拼接：图片 token 放在文本 token 前面，作为视觉上下文。
        combined_inputs = torch.cat([projected_image_features, text_embeddings], dim=1)
        return combined_inputs, combined_attention_mask, projected_image_features.size(1)

    def forward(self, images, input_ids=None, attention_mask=None, labels=None, texts=None):
        """训练/调试共用 forward。

        推荐训练时直接传入 dataset 已经构造好的 input_ids、attention_mask、labels；
        为了兼容原来的调试方式，也保留 texts 参数。
        """
        if texts is not None:
            # 旧接口：把整段文本都作为监督目标。正式微调请使用 dataset.py 生成 labels，
            # 因为它会屏蔽用户问题，只监督助手回答。
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

        inputs_embeds, combined_attention_mask, image_token_num = self._build_multimodal_inputs(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if labels is not None:
            labels = labels.to(self.device)
            # 图片部分不是离散词，不参与 CrossEntropyLoss，使用 -100 忽略。
            image_labels = torch.full(
                (labels.size(0), image_token_num),
                -100,
                dtype=torch.long,
                device=self.device
            )
            labels = torch.cat([image_labels, labels], dim=1)

        outputs = self.language_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels,
            use_cache=False
        )
        return outputs

    @torch.no_grad()
    def generate(self, images, prompts, max_new_tokens=128, **generation_kwargs):
        """多模态生成接口，用于训练后的简单验证。"""
        self.eval()
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
    
    
