import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model 

# ======================
# 1. 加载模型
# ======================
model_name = "models/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# ======================
# 2. 冻结视觉编码器
# ======================
for param in model.vision_model.parameters():
    param.requires_grad = False
    
# debug: 打印模型结构，确认视觉编码器部分被冻结，查看文本部分名字用于挂载 LoRA
for name, module in model.named_modules():
    print(name)
    
print("Vision encoder frozen ✅")

# ======================
# 3. 配置 LoRA（只作用在文本部分）
# ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "query", "value",  # attention层
    ],
    
    # 这里不能设置task_type，因为 虽然 BLIP 的文本部分是 encoder-decoder 结构，但是他是多模态模型，输入有pixcel_val 和 input_ids,
    # 不能直接当作语言模型训练
    # task_type后peft会自动在forward时构建一个inputs_embeds,和BlipForConditionalGeneration的输入不一致了。
    lora_dropout=0.05,
    bias="none",
    # task_type="SEQ_2_SEQ_LM" 
    
)


model = get_peft_model(model, lora_config)

# debug: 打印可训练参数，确认 LoRA 已正确挂载在文本部分的 attention 层
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        
model.print_trainable_parameters()

# ======================
# 4. 构造数据集
# ======================

class CocoCaptionDataset(Dataset):
    def __init__(self, annotation_file, image_dir, processor):
        """
        annotation_file: captions_val2017.json
        image_dir: val2017/
        """
        with open(annotation_file, 'r') as f:
            coco = json.load(f)

        self.image_dir = image_dir
        self.processor = processor

        # 构建 image_id -> file_name
        self.id2file = {img["id"]: img["file_name"] for img in coco["images"]}

        # 展平 annotations（每条 caption 作为一个样本）
        self.samples = []
        for ann in coco["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            file_name = self.id2file[image_id]

            self.samples.append((file_name, caption))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, caption = self.samples[idx]

        img_path = os.path.join(self.image_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # 将padding token的 label 设置为 -100，避免计算 loss 时对 padding 部分进行梯度更新
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        # debug: 打印输入的 shapes，确认视觉特征和文本输入正确
        # for k,v in inputs.items():
        #     print(k,inputs[k].shape)
        
        return inputs
        

dataset = CocoCaptionDataset("dataset/COCOCaption/annotations/captions_val2017.json", "dataset/COCOCaption/val2017", processor)

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True) 在trainer中会自动处理 dataloader，无需手动创建

# ======================
# 5. 训练
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) trainer会自动创建优化器，无需手动创建

model.train() 

# 训练参数配置
training_args = TrainingArguments(
    output_dir="week05_blip_blip2_caption/sft_blip_outputs",
    num_train_epochs=3,
    per_device_train_batch_size=10,
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-5,

    fp16=True,  # 或 bf16=True（如果支持）
    gradient_accumulation_steps=1,
)

# Trainer 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer, 
    data_collator=default_data_collator,
)

# 开始训练
trainer.train()

# ======================
# 6. 保存 LoRA 权重
# ======================
model.save_pretrained("week05_blip_blip2_caption/sft_blip_lora")