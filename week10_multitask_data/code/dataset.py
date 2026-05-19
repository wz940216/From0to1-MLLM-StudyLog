import json
import os
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import ConcatDataset, Dataset



def _read_json(path):
    """读取 LLaVA-CC3M 的 chat.json 标注文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_text(text):
    """保留 <image> 占位符，只清理首尾空白。"""
    return text.strip()


def build_prompt(question):
    """构造接近 LLaVA 的单轮 USER/ASSISTANT 模板。

    <image> 会作为特殊 token 保留在文本中，模型侧会在该位置插入视觉 patch。
    """
    question = _clean_text(question)
    return f"USER: {question}\nASSISTANT: "


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

    def __init__(self, dataset_path, image_dir, annotation_file, max_samples=None, task_name=None):
        self.dataset_path = dataset_path
        self.image_dir = os.path.join(self.dataset_path, image_dir)
        self.task_name = task_name or os.path.basename(os.path.normpath(dataset_path))
        annotation_path = os.path.join(self.dataset_path, annotation_file)
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
            "image_path": image_path,
            "task_name": self.task_name
        }


class MultiTaskLlavaDataset(Dataset):
    """把多个 LLaVA 风格数据集合并成一个训练集。

    该类只负责把多个子数据集拼接起来；真正的按比例采样交给
    train.py 中的 WeightedRandomSampler，这样数据读取和采样策略保持解耦。
    """

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
        """为 WeightedRandomSampler 构造逐样本权重。

        每个任务内所有样本共享同一权重：任务采样率 / 任务样本数。
        因此按 replacement 抽样时，抽中某个任务的总概率接近 SAMPLE_RATE。
        """
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
    """把原始样本拼成可训练 batch。

    labels 的关键规则：
    - prompt 部分是用户问题和“助手：”前缀，只作为条件输入，不计算 loss。
    - answer 部分是模型需要学习生成的目标，保留真实 token id。
    - padding 部分也置为 -100，避免 padding token 参与 loss。
    """

    tokenizer: object
    max_length: int = 512

    def __post_init__(self):
        if self.tokenizer is None:
            raise ValueError("LlavaCollator 需要传入 tokenizer，不能为 None。")

    def __call__(self, features):
        images = [x["image"] for x in features]
        prompts = [x["prompt"] for x in features]
        answers = [x["answer"] for x in features]
        task_names = [x.get("task_name", "unknown") for x in features]

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
            "labels": labels,
            "task_names": task_names
        }


if __name__ == "__main__":
    dataset = LlavaPretrainDataset(
        dataset_path="dataset/LLaVA-CC3M-Pretrain-595K",
        image_dir="images",
        annotation_file="chat.json",
        max_samples=1
    )
    sample = dataset[0]
    print(sample.keys())
    print(sample["image"], sample["prompt"], sample["answer"])
