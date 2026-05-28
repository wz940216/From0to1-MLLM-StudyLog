import json
import os
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"


def _read_json(path):
    """读取图文对话 JSON 标注文件。

    当前 Dataset 同时兼容两种格式：

    1. 旧 LLaVA conversations 格式：
       {
         "image": "xxx.jpg",
         "conversations": [
           {"from": "human", "value": "<image>\n问题"},
           {"from": "gpt", "value": "答案"}
         ]
       }

    2. OpenAI chat messages 格式：
       {
         "messages": [
           {
             "role": "user",
             "content": [
               {"type": "image", "image": "val2017/xxx.jpg"},
               {"type": "text", "text": "图中有什么？"}
             ]
           },
           {"role": "assistant", "content": "图中是..."}
         ]
       }

    训练内部仍会把第一轮 user 文本还原成带 <image> 的 prompt，因为当前
    MiniLLaVA 模型通过 <image> token 的位置插入视觉特征。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_text(text):
    """清理文本首尾空白。"""
    return str(text).strip()


def _strip_image_token(text):
    """OpenAI chat 的文本 content 不应包含 <image>，读取时需要去掉重复占位符。"""
    return _clean_text(text).replace(IMAGE_TOKEN, "").strip()


def _ensure_image_token(text):
    """把第一轮问题转换成当前 MiniLLaVA 训练需要的 <image> + 文本形式。"""
    text = _clean_text(text)
    if IMAGE_TOKEN in text:
        return text
    return f"{IMAGE_TOKEN}\n{text}"


def build_prompt(question):
    """构造单轮推理 prompt，供 infer.py 继续使用。"""
    question = _clean_text(question)
    return f"USER: {question}\nASSISTANT: "


def _extract_openai_user_content(content):
    """从 OpenAI chat user content 中取出文本和图片路径。

    OpenAI 多模态格式里，user content 既可能是字符串，也可能是一个 block 列表：
    - 字符串：后续纯文本追问，例如 "他穿什么颜色衣服？"
    - block 列表：第一轮图文输入，例如 image block + text block

    返回值是 (question_text, image_ref)。如果该轮没有图片，image_ref 为 None。
    """
    if isinstance(content, str):
        return _clean_text(content), None

    if not isinstance(content, list):
        return _clean_text(content), None

    text_parts = []
    image_ref = None
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(_clean_text(block.get("text", "")))
        elif block_type == "image":
            image_ref = block.get("image") or block.get("url") or block.get("path")

    question = "\n".join(part for part in text_parts if part).strip()
    return question, image_ref


def extract_openai_chat_turns(messages):
    """把 OpenAI chat messages 解析成训练用 turns，并返回图片引用。

    只监督 assistant 回复：user 消息进入上下文，assistant 消息作为 label。
    第一轮带图片的 user 消息会被还原成 "<image>\n问题"，这样模型前向时
    _build_multimodal_inputs() 能找到 <image> token 并插入视觉 patch embedding。
    """
    turns = []
    pending_question = None
    image_ref = None

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            question, current_image_ref = _extract_openai_user_content(content)
            if current_image_ref and image_ref is None:
                image_ref = current_image_ref
            if not question:
                continue

            # 只有第一张图对应当前样本的视觉输入；后续追问不再重复 image block。
            if current_image_ref or (image_ref is not None and not turns and pending_question is None):
                pending_question = _ensure_image_token(_strip_image_token(question))
            else:
                pending_question = _strip_image_token(question)
            continue

        if role == "assistant" and pending_question is not None:
            answer = _clean_text(content)
            if answer:
                turns.append({
                    "question": pending_question,
                    "answer": answer,
                })
            pending_question = None

    if not turns:
        raise ValueError("OpenAI messages 缺少完整 user/assistant 对话轮次，无法构造监督数据。")
    if image_ref is None:
        raise ValueError("OpenAI messages 第一轮 user content 缺少 image block，无法定位图片。")
    return turns, image_ref


def extract_llava_dialogue_turns(conversations):
    """把旧 LLaVA conversations 解析成按顺序排列的多轮 QA。"""
    turns = []
    pending_question = None

    for message in conversations:
        role = message.get("from")
        value = _clean_text(message.get("value", ""))
        if not value:
            continue

        if role == "human":
            pending_question = value
            continue

        if role == "gpt" and pending_question is not None:
            turns.append({
                "question": pending_question,
                "answer": value,
            })
            pending_question = None

    if not turns:
        raise ValueError("样本缺少完整 human/gpt 对话轮次，无法构造监督数据。")
    return turns


def resolve_image_path(dataset_path, image_dir, item, image_ref=None):
    """根据样本格式解析真实图片路径。

    - OpenAI messages 格式优先使用 image block 中的 image 字段。
      如果它写的是 val2017/xxx.jpg，就相对 dataset_path 解析。
    - 旧 LLaVA 格式使用顶层 item["image"]，并拼接配置里的 image_dir。
    """
    if image_ref:
        image_ref = str(image_ref)
        if os.path.isabs(image_ref):
            return image_ref
        return os.path.join(dataset_path, image_ref)

    if "image" not in item:
        raise ValueError("样本缺少 image 字段，且 messages 中也没有 image block。")
    return os.path.join(dataset_path, image_dir, item["image"])


def build_training_segments(turns, eos_token=""):
    """把多轮 QA 转成文本片段，并标记哪些片段需要计算 loss。

    多轮结构是：
    USER: 第 1 轮问题
    ASSISTANT: 第 1 轮答案
    USER: 第 2 轮问题
    ASSISTANT: 第 2 轮答案

    训练时只让 assistant 答案参与 loss；USER 问题和角色前缀只作为条件输入。
    """
    segments = []
    for turn in turns:
        question = _clean_text(turn["question"])
        answer = _clean_text(turn["answer"])

        segments.append({
            "text": f"USER: {question}\nASSISTANT: ",
            "train": False,
        })
        segments.append({
            "text": answer + eos_token + "\n",
            "train": True,
        })
    return segments


class LlavaPretrainDataset(Dataset):
    """读取图文对话数据的 Dataset。

    这个类现在同时支持旧 LLaVA conversations 和 OpenAI chat messages。
    每条样本只返回 PIL 图片和结构化 turns；tokenize、padding、labels mask 都在
    LlavaCollator 里完成。
    """

    def __init__(self, dataset_path, image_dir, annotation_file, max_samples=None, task_name=None):
        self.dataset_path = dataset_path
        self.image_dir = image_dir
        self.task_name = task_name or os.path.basename(os.path.normpath(dataset_path))
        annotation_path = os.path.join(self.dataset_path, annotation_file)
        self.samples = _read_json(annotation_path)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        if "messages" in item:
            turns, image_ref = extract_openai_chat_turns(item["messages"])
            image_path = resolve_image_path(self.dataset_path, self.image_dir, item, image_ref=image_ref)
        else:
            turns = extract_llava_dialogue_turns(item["conversations"])
            image_path = resolve_image_path(self.dataset_path, self.image_dir, item)

        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "turns": turns,
            "image_path": image_path,
            "task_name": self.task_name,
            "sample_id": item.get("id", str(index)),
        }


class MultiTaskLlavaDataset(Dataset):
    """把多个 LLaVA/OpenAI chat 风格数据集合并成一个训练集。"""

    def __init__(self, datasets, task_names=None, sample_rates=None):
        if not datasets:
            raise ValueError("MultiTaskLlavaDataset 至少需要一个子数据集。")

        self.datasets = list(datasets)
        self.task_names = task_names or [
            getattr(dataset, "task_name", f"task_{idx}")
            for idx, dataset in enumerate(self.datasets)
        ]
        self.sample_rates = sample_rates
        self.concat_dataset = ConcatDataset(self.datasets)
        self.sample_to_task = []
        for task_idx, dataset in enumerate(self.datasets):
            self.sample_to_task.extend([task_idx] * len(dataset))

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, index):
        sample = self.concat_dataset[index]
        task_idx = self.sample_to_task[index]
        sample.setdefault("task_name", self.task_names[task_idx])
        sample["task_index"] = task_idx
        return sample

    def build_sample_weights(self):
        """为 WeightedRandomSampler 构造逐样本权重。"""
        if self.sample_rates is None:
            return None
        if len(self.sample_rates) != len(self.datasets):
            raise ValueError("SAMPLE_RATE 数量必须和数据集数量一致。")

        total_rate = sum(float(rate) for rate in self.sample_rates)
        if total_rate <= 0:
            raise ValueError("SAMPLE_RATE 之和必须大于 0。")

        weights = []
        for dataset, rate in zip(self.datasets, self.sample_rates):
            dataset_len = len(dataset)
            if dataset_len == 0:
                raise ValueError("子数据集不能为空。")
            task_weight = float(rate) / total_rate / dataset_len
            weights.extend([task_weight] * dataset_len)
        return weights


@dataclass
class LlavaCollator:
    """把 Dataset 返回的原始样本拼成可训练 batch。

    labels 构造规则：
    - USER 问题、"USER:"、"ASSISTANT:" 前缀都只是条件输入，label 置为 -100。
    - 每一轮 assistant 答案都是训练目标，label 保留真实 token id。
    - eos token 也作为答案结束标记参与训练。
    - padding 位置置为 -100，避免 padding token 影响 loss。
    """

    tokenizer: object
    max_length: int = 512

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("LlavaCollator 需要传入 tokenizer，不能为 None。")

    def _tokenize_segment(self, text):
        """tokenize 单个文本片段，不额外插入 BOS/EOS。"""
        return self.tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids

    def _encode_one_sample(self, turns):
        """把一条样本的多轮 turns 编码成 input_ids 和 labels。"""
        eos = self.tokenizer.eos_token or ""
        segments = build_training_segments(turns, eos_token=eos)

        input_ids = []
        labels = []
        for segment in segments:
            token_ids = self._tokenize_segment(segment["text"])
            input_ids.extend(token_ids)

            if segment["train"]:
                labels.extend(token_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(token_ids))

        # 当前保持和原项目一致的右截断策略。后续如果对话很长，可以改成
        # “保留第一轮 <image> + 最近若干轮”的上下文截断策略。
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, labels

    def __call__(self, features):
        images = [x["image"] for x in features]
        task_names = [x.get("task_name", "unknown") for x in features]
        sample_ids = [x.get("sample_id", "") for x in features]

        encoded = [self._encode_one_sample(x["turns"]) for x in features]
        input_id_lists, _, _ = zip(*encoded)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer 既没有 pad_token_id 也没有 eos_token_id，无法 padding。")

        batch_max_len = max(len(ids) for ids in input_id_lists)
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for input_ids, attention_mask, labels in encoded:
            pad_len = batch_max_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_token_id] * pad_len)
            padded_attention_masks.append(attention_mask + [0] * pad_len)
            padded_labels.append(labels + [IGNORE_INDEX] * pad_len)

        return {
            "images": images,
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "task_names": task_names,
            "sample_ids": sample_ids,
        }


if __name__ == "__main__":
    dataset = LlavaPretrainDataset(
        dataset_path="dataset/COCOCaption",
        image_dir="val2017",
        annotation_file="annotations/captions_val2017_multiturn.json",
        max_samples=1,
    )
    sample = dataset[0]
    print(sample.keys())
    print(sample["image_path"])
    print(sample["turns"])
