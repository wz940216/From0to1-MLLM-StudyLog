# week14_dialogue_stability_output_control

本周在 week13 的 OpenAI Chat 三角色对齐基础上，补充多轮对话稳定性和结构化输出控制。

## 目标

- 长对话只保留必要上下文，降低 prompt 过长、遗忘图片信息和角色跑偏的风险。
- 针对 JSON 业务输出增加 few-shot Prompt 模板，让模型看到明确的字段和示例。
- 封装输出解析模块，统一完成 JSON 提取、校验和失败重试。
- 编写固定多轮对话测试脚本，自动喂问题并把每轮结果保存为 JSONL。

## 对话截断策略

推理侧在 `infer.py` 中使用 `--max-history-turns` 控制历史轮数，默认保留最近 3 轮。

```python
def truncate_history(history, max_history_turns=None):
    """保留最近 N 轮 user/assistant 历史，降低长对话跑偏风险。"""
    if max_history_turns is None or int(max_history_turns) <= 0:
        return list(history)
    return list(history)[-int(max_history_turns) * 2:]
```

训练侧在 `dataset.py` 中使用两级截断：

```text
轮次级：如果样本轮数 > N，保留第 1 轮图文输入 + 最近 N-1 轮。
Token 级：如果文本 token 仍超过 MAX_LENGTH，优先保留最后一段 assistant answer，剩余预算再从最近上下文向前填充。
```


```python
def _crop_segment_keep_image(self, segment, budget, image_token_id):
        """裁剪包含 <image> 的片段，至少保留图片 token。"""
        if budget <= 0:
            return None
        input_ids = segment["input_ids"]
        labels = segment["labels"]
        if image_token_id not in input_ids:
            return {
                "input_ids": input_ids[-budget:],
                "labels": labels[-budget:],
                "train": segment["train"],
            }

        image_pos = input_ids.index(image_token_id)
        if len(input_ids) <= budget:
            return segment

        end = min(len(input_ids), image_pos + budget)
        start = max(0, end - budget)
        if not (start <= image_pos < end):
            start = image_pos
            end = min(len(input_ids), image_pos + budget)
        return {
            "input_ids": input_ids[start:end],
            "labels": labels[start:end],
            "train": segment["train"],
        }

    def _fit_segments_to_budget(self, encoded_segments):
        """按 token 预算裁剪：保留 <image>，优先保留最后 assistant answer。"""
        max_length = int(self.max_length)
        if max_length <= 1:
            raise ValueError("max_length 必须大于 1，至少要容纳 <image> 和一个训练 token。")

        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        total_len = sum(len(segment["input_ids"]) for segment in encoded_segments)
        if total_len <= max_length:
            return encoded_segments

        image_idx = None
        for idx, segment in enumerate(encoded_segments):
            if image_token_id in segment["input_ids"]:
                image_idx = idx
                break
        if image_idx is None:
            raise ValueError("样本缺少 <image> token，无法构造多模态输入。")

        last_train_idx = None
        for idx in range(len(encoded_segments) - 1, -1, -1):
            if encoded_segments[idx]["train"]:
                last_train_idx = idx
                break
        if last_train_idx is None:
            raise ValueError("样本没有可训练的 assistant answer。")

        image_anchor = self._crop_segment_keep_image(encoded_segments[image_idx], 1, image_token_id)
        last_answer = encoded_segments[last_train_idx]
        answer_budget = max_length - len(image_anchor["input_ids"])
        answer_ids = last_answer["input_ids"]
        answer_labels = last_answer["labels"]
        if len(answer_ids) > answer_budget:
            return [
                image_anchor,
                {
                    "input_ids": answer_ids[-answer_budget:],
                    "labels": answer_labels[-answer_budget:],
                    "train": True,
                },
            ]

        selected_middle = []
        used = len(image_anchor["input_ids"]) + len(answer_ids)
        full_image_segment = encoded_segments[image_idx]

        for idx in range(last_train_idx - 1, -1, -1):
            remain = max_length - used
            if remain <= 0:
                break

            segment = encoded_segments[idx]
            if idx == image_idx:
                cropped_image = self._crop_segment_keep_image(segment, remain + len(image_anchor["input_ids"]), image_token_id)
                used += len(cropped_image["input_ids"]) - len(image_anchor["input_ids"])
                image_anchor = cropped_image
                break

            seg_len = len(segment["input_ids"])
            if seg_len <= remain:
                selected_middle.insert(0, segment)
                used += seg_len
                continue

            selected_middle.insert(0, {
                "input_ids": segment["input_ids"][-remain:],
                "labels": segment["labels"][-remain:],
                "train": segment["train"],
            })
            break

        return [image_anchor] + selected_middle + [last_answer]

    def _flatten_segments(self, encoded_segments):
        input_ids = []
        labels = []
        for segment in encoded_segments:
            input_ids.extend(segment["input_ids"])
            labels.extend(segment["labels"])
        if self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) not in input_ids:
            raise ValueError("截断后样本缺少 <image> token，请增大 MAX_LENGTH 或减小 MAX_TURNS。")
        return input_ids, labels
```

这样既能保留图片进入模型的位置，也能避免简单 `[:MAX_LENGTH]` 把尾部 answer 训练目标截掉。预算极小时会至少保留一个 `<image>` token 和最后 answer 的尾部；如果 `MAX_LENGTH <= 1`，collator 会直接报错。

## JSON few-shot 模板

`prompt_templates.py` 中提供 `JSON_FEW_SHOT_PROMPT`，固定输出字段：

```json
{"answer": "...", "confidence": "...", "evidence": "..."}
```

推理时当前用户问题会被拼到 few-shot 模板之后，要求模型只输出合法 JSON 对象。

## 输出解析和失败重试

`output_parser.py` 提供：

- `normalize_model_text()`：清理 prompt 回显、角色前缀和 markdown 代码块。
- `extract_json_candidate()`：从模型输出中截取 JSON 对象。
- `parse_json_output()`：校验 JSON 顶层对象和必需字段。
- `JSON_REPAIR_PROMPT`：当 JSON 校验失败时，触发一次重新生成。

```python
import json
import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    ok: bool
    data: object = None
    text: str = ""
    error: str = ""


JSON_REPAIR_PROMPT = """上一轮回答不是合法 JSON。请只根据原始问题重新输出一个合法 JSON 对象。
不要输出 markdown 代码块，不要输出解释文字。
固定格式：
{"answer": "这里填写对用户问题的回答"}"""


def strip_code_fence(text):
    text = str(text).strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_candidate(text):
    """从模型输出中提取最可能的 JSON 对象片段。"""
    text = strip_code_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return text.strip()
    return text[start:end + 1].strip()


def parse_json_output(text, required_keys=None):
    """校验 JSON 输出，并可选检查必需字段。"""
    candidate = extract_json_candidate(text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParseResult(False, text=candidate, error=str(exc))

    if not isinstance(data, dict):
        return ParseResult(False, data=data, text=candidate, error="JSON 顶层必须是对象。")

    missing = [key for key in (required_keys or []) if key not in data]
    if missing:
        return ParseResult(False, data=data, text=candidate, error=f"JSON 缺少字段: {', '.join(missing)}")

    return ParseResult(True, data=data, text=json.dumps(data, ensure_ascii=False))


def normalize_model_text(text, prompt=""):
    """尽量只保留当前轮 assistant 的回答。"""
    text = str(text).strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()

    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    if "USER:" in text:
        text = text.split("USER:", 1)[0].strip()

    return strip_code_fence(text)


def looks_like_json_request(text):
    return re.search(r"\bjson\b|结构化|字段|格式", str(text), flags=re.IGNORECASE) is not None

```

`infer.py` 的 `answer_one_turn()` 默认会校验 `answer` 字段；失败后自动追加修复提示重试一次。

```python
def answer_one_turn(model, image, history, question, gen_config, system_prompt=None, max_history_turns=None, retry_on_json_error=True):
    """执行一轮带上下文的图文对话推理，并返回 assistant 回答。"""
    prompt = build_context_prompt(
        history,
        question,
        system_prompt=system_prompt,
        max_history_turns=max_history_turns,
    )
    answer = _generate_text(model, image, prompt, gen_config)
    parsed = parse_json_output(answer, required_keys=["answer"])
    if parsed.ok:
        return parsed.text
    if not retry_on_json_error:
        return answer

    retry_question = f"{clean_user_text(question)}\n\n{JSON_REPAIR_PROMPT}\n上一轮错误：{parsed.error}"
    retry_prompt = build_context_prompt(
        history,
        retry_question,
        system_prompt=system_prompt,
        max_history_turns=max_history_turns,
    )
    retry_answer = _generate_text(model, image, retry_prompt, gen_config)
    retry_parsed = parse_json_output(retry_answer, required_keys=["answer"])
    return retry_parsed.text if retry_parsed.ok else retry_answer
```
## 多轮对话测试

固定测试脚本：

```shell
python week14_dialogue_stability_output_control/code/test_multiturn_dialogue.py \
  --config week14_dialogue_stability_output_control/configs/caption_only_cpu.yaml \
  --checkpoint none \
  --max-history-turns 3
```

输出默认写入：

```text
week14_dialogue_stability_output_control/outputs/logs/multiturn_test.jsonl
```

每条日志包含 turn、question、answer、json_ok、json_error 和截断后的 history 轮数，便于比较不同 `--max-history-turns` 下的稳定性。

## 训练

```shell
accelerate launch --multi_gpu week14_dialogue_stability_output_control/code/train.py \
  --config week14_dialogue_stability_output_control/configs/multitask_balanced.yaml
```

## 推理

```shell
python week14_dialogue_stability_output_control/code/infer.py \
  --config week14_dialogue_stability_output_control/configs/config.yaml \
  --checkpoint week14_dialogue_stability_output_control/outputs/checkpoints/step_2109.pt \
  --image dataset/coco128/images/train2017/000000000025.jpg \
  --question "请用 JSON 描述这张图片" \
  --question "继续用 JSON 说明判断依据" \
  --max-history-turns 3
```
