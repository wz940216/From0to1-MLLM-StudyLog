import json
import os
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset



def _read_json(path):
    """读取 LLaVA-CC3M 的 chat.json 标注文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_text(text):
    """去掉 LLaVA 数据中用于占位图片的 <image> 标记。"""
    return text.replace("<image>", "").strip()


def build_prompt(question):
    """构造训练和推理保持一致的文本模板。

    这里用简单清晰的 Q/A 模板，便于理解。真正大规模训练时也可以换成
    Qwen chat template，但要保证训练和推理使用同一套格式。
    """
    question = _clean_text(question)
    return f"用户：{question}\n助手："


def extract_qa(conversations):
    """从 LLaVA 风格 conversations 中提取第一轮 human/gpt 问答。"""
    question = None
    answer = None
    for message in conversations:
        role = message.get("from")
        value = message.get("value", "")
        if role == "human" and question is None:
            question = value
        elif role == "gpt" and answer is None:
            answer = value
        if question is not None and answer is not None:
            break
    if question is None or answer is None:
        raise ValueError("样本缺少 human/gpt 对话轮次，无法构造监督数据。")
    return question, answer.strip()


class LlavaPretrainDataset(Dataset):
    """读取 dataset/LLaVA-CC3M-Pretrain-595K/chat.json 的 Dataset。

    每条样本返回 PIL 图片、prompt 和 answer。tokenize 放在 collate_fn 中做，
    因为 batch 内需要统一 padding，放在 collate 阶段更自然。
    """

    def __init__(self, dataset_path, image_dir, annotation_file, max_samples=None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, image_dir)
        annotation_path = os.path.join(dataset_path, annotation_file)
        self.samples = _read_json(annotation_path)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        question, answer = extract_qa(item["conversations"])
        return {
            "image": image,
            "prompt": build_prompt(question),
            "answer": answer,
            "image_path": image_path
        }


@dataclass
class LlavaCollator:
    """把原始样本拼成可训练 batch。

    labels 的关键规则：
    - prompt 部分是用户问题和“助手：”前缀，只作为条件输入，不计算 loss。
    - answer 部分是模型需要学习生成的目标，保留真实 token id。
    - padding 部分也置为 -100，避免 padding token 参与 loss。
    """

    tokenizer: object
    max_length: int = 512

    def __call__(self, features):
        images = [x["image"] for x in features]
        prompts = [x["prompt"] for x in features]
        answers = [x["answer"] for x in features]

        # eos 可以明确告诉模型回答结束；如果 tokenizer 没有 eos，就退化为空字符串。
        eos = self.tokenizer.eos_token or ""
        full_texts = [prompt + answer + eos for prompt, answer in zip(prompts, answers)]

        tokenized = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = tokenized.input_ids.clone()

        # 逐条计算 prompt token 长度，并把 prompt 位置 label 屏蔽为 -100。
        for row, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True
            ).input_ids
            prompt_len = min(len(prompt_ids), labels.size(1))
            labels[row, :prompt_len] = -100

        # padding 不参与训练损失。
        labels[tokenized.attention_mask == 0] = -100

        return {
            "images": images,
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": labels
        }
