# week14_dialogue_stability_output_control

本周在 week13 的 OpenAI Chat 三角色对齐基础上，补充多轮对话稳定性和结构化输出控制。

## 目标

- 长对话只保留必要上下文，降低 prompt 过长、遗忘图片信息和角色跑偏的风险。
- 针对 JSON 业务输出增加 few-shot Prompt 模板，让模型看到明确的字段和示例。
- 封装输出解析模块，统一完成 JSON 提取、校验和失败重试。
- 编写固定多轮对话测试脚本，自动喂问题并把每轮结果保存为 JSONL。

## 对话截断策略

推理侧在 `infer.py` 中使用 `--max-history-turns` 控制历史轮数，默认保留最近 3 轮。

训练侧在 `dataset.py` 中使用两级截断：

```text
轮次级：如果样本轮数 > N，保留第 1 轮图文输入 + 最近 N-1 轮。
Token 级：如果文本 token 仍超过 MAX_LENGTH，优先保留最后一段 assistant answer，剩余预算再从最近上下文向前填充。
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

`infer.py` 的 `answer_one_turn()` 默认会校验 `answer` 字段；失败后自动追加修复提示重试一次。

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
