import argparse
import glob
import json
import os
import re
from datetime import datetime
from io import BytesIO

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import IMAGE_TOKEN, LlavaCollator, _strip_image_token
from mini_llava import MiniLlavaModel
from train import append_jsonl, build_optimizer, build_scheduler, load_config, save_checkpoint, set_seed


class LlavaDPODataset(Dataset):
    """MiniLLaVA DPO 偏好数据集。

    支持两类数据来源：
    1. 普通 JSON/JSONL：每条样本已经是 image/prompt/chosen/rejected。
    2. VLFeedback：每条样本是 image/prompt/completions，脚本会自动把评分最高的
       completion 转成 chosen，把评分最低的 completion 转成 rejected。

    VLFeedback 的 completion 有 Helpfulness、Ethical Considerations、Visual Faithfulness
    三个 Rating。这里用三项评分求和作为总分，工程上先得到一个简单可用的 DPO 偏好对。
    """

    def __init__(self, data_file=None, image_root=None, max_samples=None, example_image=None, toy_data=False):
        self.image_root = image_root
        self.is_vlfeedback = False

        if toy_data or data_file is None:
            self.samples = self._build_toy_samples(example_image)
        elif self._is_vlfeedback_source(data_file):
            self.samples = self._load_vlfeedback(data_file)
            self.is_vlfeedback = True
        else:
            self.samples = self._read_preference_file(data_file)

        if max_samples is not None:
            self.samples = self.samples.select(range(min(int(max_samples), len(self.samples)))) if self.is_vlfeedback else self.samples[:max_samples]

    def _is_vlfeedback_source(self, path):
        path = str(path)
        if path == "MMInstruction/VLFeedback":
            return True
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "README.md")):
            with open(os.path.join(path, "README.md"), "r", encoding="utf-8") as f:
                return "VLFeedback" in f.read(2048)
        return False

    def _load_vlfeedback(self, path):
        """加载本地或 Hub 上的 VLFeedback。

        本地目录 dataset/VLFeedback 中已有 9 个 parquet shard 时，传入该目录即可。
        如果要从 Hub 加载，也可以传入 MMInstruction/VLFeedback。
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("加载 VLFeedback 需要安装 datasets：pip install datasets") from exc

        try:
            dataset = load_dataset(path, split="train")
        except Exception:
            parquet_files = sorted(glob.glob(os.path.join(str(path), "data", "train-*.parquet")))
            if not parquet_files:
                raise
            dataset = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
        if len(dataset) == 0:
            raise ValueError(f"VLFeedback 数据集为空: {path}")
        return dataset

    def _read_preference_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
            return data["data"] if isinstance(data, dict) and "data" in data else data

    def _build_toy_samples(self, example_image):
        image = example_image or "dataset/coco128/images/train2017/000000000025.jpg"
        return [
            {
                "image": image,
                "prompt": "请描述图中的主要内容。",
                "chosen": "图中包含一个清晰可见的室内场景，回答应围绕图片中能观察到的物体和关系展开。",
                "rejected": "这张图展示了与画面无关的危险操作步骤。",
            },
            {
                "image": image,
                "prompt": "如果用户让你根据图片编造不存在的细节，你应该怎么回答？",
                "chosen": "我应该只描述图中能确认的信息，并说明无法确定的部分，避免编造。",
                "rejected": "我可以随便补充看起来合理但图片里没有的信息。",
            },
        ]

    def _resolve_image_path(self, image_path):
        image_path = str(image_path)
        if os.path.isabs(image_path):
            return image_path
        if self.image_root:
            return os.path.join(self.image_root, image_path)
        return image_path

    def _load_image(self, image_value):
        """把普通路径或 HF datasets Image 字段统一转成 RGB PIL.Image。"""
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if isinstance(image_value, dict):
            if image_value.get("bytes") is not None:
                return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
            if image_value.get("path"):
                return Image.open(self._resolve_image_path(image_value["path"])).convert("RGB")
        return Image.open(self._resolve_image_path(image_value)).convert("RGB")

    def _rating_to_score(self, rating):
        """把 VLFeedback 中的 Rating 字符串转成数字分数。"""
        if rating is None:
            return 0.0
        match = re.search(r"-?\d+(?:\.\d+)?", str(rating))
        return float(match.group(0)) if match else 0.0

    def _completion_score(self, completion):
        annotations = completion.get("annotations") or {}
        score = 0.0
        for key in ["Helpfulness", "Ethical Considerations", "Visual Faithfulness"]:
            score += self._rating_to_score((annotations.get(key) or {}).get("Rating"))
        return score

    def _normalize_completions(self, completions):
        """兼容 VLFeedback 的两种 completion 表示。

        从 datasets 读 parquet 时，sequence struct 会变成 dict-of-list：
        {"annotations": [...], "model": [...], "response": [...]}。
        旧 JSONL 样本通常是 list-of-dict。这里统一成 list-of-dict。
        """
        if isinstance(completions, dict):
            responses = completions.get("response", [])
            models = completions.get("model", [""] * len(responses))
            annotations = completions.get("annotations", [{}] * len(responses))
            return [
                {
                    "response": response,
                    "model": models[idx] if idx < len(models) else "",
                    "annotations": annotations[idx] if idx < len(annotations) else {},
                }
                for idx, response in enumerate(responses)
            ]
        return list(completions or [])

    def _convert_vlfeedback_item(self, item):
        completions = [
            c
            for c in self._normalize_completions(item.get("completions", []))
            if str(c.get("response", "")).strip()
        ]
        if len(completions) < 2:
            raise ValueError(f"VLFeedback 样本缺少至少两个有效 completion: {item.get('id')}")

        ranked = sorted(completions, key=self._completion_score, reverse=True)
        chosen = ranked[0]
        rejected = ranked[-1]
        return {
            "image": item["image"],
            "prompt": item["prompt"],
            "chosen": chosen["response"],
            "rejected": rejected["response"],
            "system_prompt": str(item.get("system", "")).strip(),
            "sample_id": item.get("id", ""),
            "chosen_model": chosen.get("model", ""),
            "rejected_model": rejected.get("model", ""),
            "chosen_score": self._completion_score(chosen),
            "rejected_score": self._completion_score(rejected),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raw_item = self.samples[index]
        item = self._convert_vlfeedback_item(raw_item) if self.is_vlfeedback else raw_item
        image = self._load_image(item["image"])
        prompt = _strip_image_token(item["prompt"])
        system_prompt = str(item.get("system_prompt", item.get("system", ""))).strip()

        return {
            "image": image,
            "prompt": prompt,
            "chosen": str(item["chosen"]).strip(),
            "rejected": str(item["rejected"]).strip(),
            "system_prompt": system_prompt,
            "sample_id": item.get("sample_id", item.get("id", str(index))),
            "chosen_model": item.get("chosen_model", ""),
            "rejected_model": item.get("rejected_model", ""),
            "chosen_score": item.get("chosen_score"),
            "rejected_score": item.get("rejected_score"),
        }

def save_example_dataset(path, example_image=None):
    """保存一份 DPO JSONL 样例，便于按字段准备自己的偏好数据。"""
    dataset = LlavaDPODataset(example_image=example_image, toy_data=True)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in dataset.samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


class LlavaDPOCollator:
    """把一批偏好样本转成 chosen/rejected 两个 MiniLLaVA batch。"""

    def __init__(self, tokenizer, max_length=512, max_turns=1):
        self.sft_collator = LlavaCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            max_turns=max_turns,
        )

    def _to_sft_feature(self, sample, answer):
        # 当前 MiniLLaVA 通过第一轮 USER 中的 <image> token 插入视觉特征。
        return {
            "image": sample["image"],
            "turns": [
                {
                    "question": f"{IMAGE_TOKEN}\n{sample['prompt']}",
                    "answer": answer,
                }
            ],
            "system_prompt": sample.get("system_prompt", ""),
            "sample_id": sample.get("sample_id", ""),
            "task_name": "dpo",
        }

    def __call__(self, features):
        chosen_features = [self._to_sft_feature(sample, sample["chosen"]) for sample in features]
        rejected_features = [self._to_sft_feature(sample, sample["rejected"]) for sample in features]
        return {
            "chosen": self.sft_collator(chosen_features),
            "rejected": self.sft_collator(rejected_features),
            "sample_ids": [sample.get("sample_id", "") for sample in features],
        }


def load_checkpoint(model, checkpoint_path):
    """加载 SFT checkpoint，DPO 通常从 SFT 模型继续训练。"""
    if checkpoint_path is None:
        return None
    state = torch.load(checkpoint_path, map_location=model.device)
    state_dict = state.get("model", state)
    model_state = model.state_dict()

    loadable_state = {}
    skipped_shape = []
    for name, value in state_dict.items():
        if name not in model_state:
            loadable_state[name] = value
            continue
        if tuple(value.shape) != tuple(model_state[name].shape):
            skipped_shape.append({
                "name": name,
                "checkpoint_shape": tuple(value.shape),
                "model_shape": tuple(model_state[name].shape),
            })
            continue
        loadable_state[name] = value

    incompatible = model.load_state_dict(loadable_state, strict=False)
    return {
        "path": checkpoint_path,
        "loaded_keys": list(loadable_state.keys()),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "skipped_shape": skipped_shape,
    }


def freeze_model(model):
    """冻结 reference model；DPO 中 reference 只提供基准 logprob，不参与更新。"""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def get_base_model(model):
    """Accelerate/DDP 包装后，取回真正的 MiniLlavaModel 以调用内部多模态展开函数。"""
    return model.module if hasattr(model, "module") else model


def sequence_log_probs(model, batch):
    """计算每条样本 answer token 的平均 log probability。

    这里不能直接用原始 labels 对齐 outputs.logits，因为 MiniLLaVA 会把一个 <image>
    token 展开成很多视觉 patch embedding。必须先拿到展开后的 combined_labels，再按
    decoder-only LM 的规则右移一位计算 answer token 的 logprob。
    """
    base_model = get_base_model(model)
    inputs_embeds, attention_mask, labels = base_model._build_multimodal_inputs(
        images=batch["images"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    outputs = base_model.language_decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )

    logits = outputs.logits[:, :-1, :]
    labels = labels[:, 1:].to(logits.device)
    loss_mask = labels != -100
    safe_labels = labels.masked_fill(~loss_mask, 0)

    token_log_probs = torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probs = token_log_probs * loss_mask

    token_count = loss_mask.sum(dim=-1).clamp(min=1)
    return token_log_probs.sum(dim=-1) / token_count


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    """DPO loss。

    policy_logratios 表示当前模型更偏向 chosen 还是 rejected。
    ref_logratios 表示参考模型原本的偏好。
    DPO 优化的是两者差值，让 policy 比 reference 更偏向 chosen。
    """
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_logratios - ref_logratios
    losses = -F.logsigmoid(float(beta) * logits)
    rewards_chosen = float(beta) * (policy_chosen_logps - ref_chosen_logps).detach()
    rewards_rejected = float(beta) * (policy_rejected_logps - ref_rejected_logps).detach()
    return losses.mean(), logits.detach(), rewards_chosen, rewards_rejected


def move_tensor_batch_to_device(batch, device):
    """PIL 图片留在 CPU；input_ids/attention_mask/labels 移到当前进程设备。"""
    moved = dict(batch)
    for key in ["input_ids", "attention_mask", "labels"]:
        moved[key] = batch[key].to(device)
    return moved


def optional_path(value):
    """把配置或命令行中的 none/null/空字符串统一转成 None。"""
    if value is None:
        return None
    if str(value).strip().lower() in {"", "none", "null"}:
        return None
    return value


def parse_args_and_config():
    """先读取 --config，再用配置里的 DPO 字段作为命令行默认值。"""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="week15_safety_alignment/configs/multitask_balanced.yaml", help="项目配置文件路径")
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    dpo_config = config.get("DPO", {})

    parser = argparse.ArgumentParser(description="MiniLLaVA DPO 偏好优化训练脚本", parents=[config_parser])
    parser.add_argument("--train-file", default=dpo_config.get("TRAIN_FILE", "dataset/VLFeedback"), help="DPO 数据来源；可用 dataset/VLFeedback、MMInstruction/VLFeedback，或普通 JSON/JSONL")
    parser.add_argument("--image-root", default=dpo_config.get("IMAGE_ROOT"), help="当 train-file 中 image 是相对路径时，用这个目录拼接")
    parser.add_argument("--checkpoint", default=dpo_config.get("CHECKPOINT"), help="可选：SFT checkpoint 路径，policy 和 reference 都从它初始化")
    parser.add_argument("--save-example-data", default=dpo_config.get("SAVE_EXAMPLE_DATA"), help="只保存一份 DPO 示例 JSONL 后退出")
    parser.add_argument("--example-image", default=dpo_config.get("EXAMPLE_IMAGE"), help="生成示例数据或无 train-file 调试时使用的图片")
    parser.add_argument("--max-samples", type=int, default=dpo_config.get("MAX_SAMPLES"), help="调试时只取前 N 条偏好数据")
    parser.add_argument("--beta", type=float, default=float(dpo_config.get("BETA", 0.1)), help="DPO beta，越大表示偏好约束越强")
    parser.add_argument("--toy-data", action=argparse.BooleanOptionalAction, default=bool(dpo_config.get("TOY_DATA", False)), help="是否不加载 VLFeedback，只使用内置两条小样本调试")
    args = parser.parse_args()

    args.train_file = optional_path(args.train_file)
    args.image_root = optional_path(args.image_root)
    args.checkpoint = optional_path(args.checkpoint)
    args.save_example_data = optional_path(args.save_example_data)
    args.example_image = optional_path(args.example_image)
    return args, config


def main():
    args, config = parse_args_and_config()

    # 当前环境中 cuDNN Conv2d 会触发 CUDNN_STATUS_NOT_INITIALIZED；CLIP patch embedding
    # 是 Conv2d，禁用 cuDNN 后仍走 CUDA 计算，能避开该后端初始化问题。
    torch.backends.cudnn.enabled = False

    if args.save_example_data:
        save_example_dataset(args.save_example_data, example_image=args.example_image)
        print(f"已保存 MiniLLaVA DPO 示例数据: {args.save_example_data}")
        return

    accelerator = Accelerator()
    set_seed(int(config["MISC"]["SEED"]))
    accelerate_set_seed(int(config["MISC"]["SEED"]))

    policy_model = MiniLlavaModel(args.config)
    ref_model = MiniLlavaModel(args.config)
    policy_load_info = load_checkpoint(policy_model, args.checkpoint)
    ref_load_info = load_checkpoint(ref_model, args.checkpoint)
    freeze_model(ref_model)
    policy_model.train()

    train_config = config["DATA"]["TRAIN_DATASET"]
    dataset = LlavaDPODataset(
        data_file=args.train_file,
        image_root=args.image_root,
        max_samples=args.max_samples,
        example_image=args.example_image,
        toy_data=args.toy_data,
    )
    collator = LlavaDPOCollator(
        tokenizer=policy_model.language_decoder.tokenizer,
        max_length=int(train_config["MAX_LENGTH"]),
        max_turns=1,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(train_config["NUM_WORKERS"]),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = build_optimizer(policy_model, config)
    num_epochs = int(config["TRAINING"]["SCHEDULER"]["NUM_EPOCHS"])
    policy_model, ref_model, optimizer, dataloader = accelerator.prepare(
        policy_model,
        ref_model,
        optimizer,
        dataloader,
    )
    total_steps = len(dataloader) * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    log_steps = int(config["TRAINING"]["LOGGING"]["LOG_STEPS"])
    log_dir = config["TRAINING"]["LOGGING"].get("LOG_DIR", "week15_safety_alignment/outputs/logs")
    train_log_path = os.path.join(log_dir, "train_dpo.jsonl")
    save_steps = int(config["TRAINING"]["CHECKPOINT"]["SAVE_STEPS"])
    save_dir = config["TRAINING"]["CHECKPOINT"]["SAVE_DIR"]
    max_norm = float(config["TRAINING"]["GRAD_CLIP"]["MAX_NORM"])

    if accelerator.is_main_process:
        if policy_load_info is not None:
            print(
                f"policy 已加载检查点: {policy_load_info['path']} "
                f"loaded={len(policy_load_info['loaded_keys'])} "
                f"missing={len(policy_load_info['missing_keys'])} "
                f"unexpected={len(policy_load_info['unexpected_keys'])} "
                f"shape_skipped={len(policy_load_info['skipped_shape'])}"
            )
            for item in policy_load_info["skipped_shape"][:20]:
                print(
                    "跳过 shape 不一致参数: "
                    f"{item['name']} checkpoint={item['checkpoint_shape']} model={item['model_shape']}"
                )
        if ref_load_info is not None and ref_load_info["skipped_shape"]:
            print(f"reference 同样跳过了 {len(ref_load_info['skipped_shape'])} 个 shape 不一致参数。")
        print(f"启用 MiniLLaVA DPO 训练: n={len(dataset)}, beta={args.beta}")
        append_jsonl(train_log_path, {
            "event": "dpo_train_start",
            "time": datetime.now().isoformat(timespec="seconds"),
            "config": args.config,
            "train_file": args.train_file,
            "checkpoint": args.checkpoint,
            "num_epochs": num_epochs,
            "total_steps": total_steps,
            "beta": args.beta,
        })

    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            chosen = move_tensor_batch_to_device(batch["chosen"], accelerator.device)
            rejected = move_tensor_batch_to_device(batch["rejected"], accelerator.device)

            policy_chosen_logps = sequence_log_probs(policy_model, chosen)
            policy_rejected_logps = sequence_log_probs(policy_model, rejected)
            with torch.no_grad():
                ref_chosen_logps = sequence_log_probs(ref_model, chosen)
                ref_rejected_logps = sequence_log_probs(ref_model, rejected)

            loss, logits, rewards_chosen, rewards_rejected = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=args.beta,
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(policy_model.parameters(), max_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
            margin_value = accelerator.gather_for_metrics(logits).mean().item()
            chosen_reward = accelerator.gather_for_metrics(rewards_chosen).mean().item()
            rejected_reward = accelerator.gather_for_metrics(rewards_rejected).mean().item()

            if global_step % log_steps == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch + 1}/{num_epochs} step={global_step} "
                    f"dpo_loss={loss_value:.4f} margin={margin_value:.4f} lr={lr:.8f}"
                )
                append_jsonl(train_log_path, {
                    "event": "dpo_train_step",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "step": global_step,
                    "loss": loss_value,
                    "margin": margin_value,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "lr": lr,
                })

            if global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(accelerator, policy_model, optimizer, scheduler, global_step, save_dir)

    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, policy_model, optimizer, scheduler, global_step, save_dir)
    if accelerator.is_main_process:
        append_jsonl(train_log_path, {
            "event": "dpo_train_end",
            "time": datetime.now().isoformat(timespec="seconds"),
            "step": global_step,
            "checkpoint_dir": save_dir,
        })


if __name__ == "__main__":
    main()
