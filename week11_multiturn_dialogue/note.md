# Multiturn Dialogue
第十一周，为模型增加多轮对话能力。之前我们的 qa 数据集主要还是用户输入图片，输入问题，模型给出回答的简单一问一答形式。而自然的对话理应是多轮的，可以随时上传图片，随时提问不同的问题，模型应该结合完整上下文进行回答。

常见的多轮对话数据格式主要有以下两种：

```json
{
  "id": "000001",
  "image": "images/000001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请描述这张图片。"
    },
    {
      "from": "gpt",
      "value": "图片中有一只橘猫趴在电脑键盘上。"
    },
    {
      "from": "human",
      "value": "猫在做什么？"
    },
    {
      "from": "gpt",
      "value": "它似乎正在睡觉。"
    }
  ]
}
```
这种格式 llava 最经典的多轮对话格式，也是我们能够从之前的 qa 版本直接扩展的格式。

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "images/demo.jpg"
        },
        {
          "type": "text",
          "text": "图中有什么？"
        }
      ]
    },
    {
      "role": "assistant",
      "content": "图中是一名正在骑自行车的人。"
    },
    {
      "role": "user",
      "content": "他穿的什么颜色衣服？"
    },
    {
      "role": "assistant",
      "content": "蓝色上衣。"
    }
  ]
}

```
还有一种是 openai 的 chat 格式，目前主流的 vlm 模型越来越趋向于这种 chat 格式。  
因为它更容易扩展视频、多图、tool call、grounding。  

既然是主流，我们这期将 dataset 中的 qa 格式转换成 openai 的 chat 格式。  
具体思路是：利用已有的 coco caotion 和 VQA 数据集中的一张图片有多个描述的和多个 QA 的特点，构造一个持续的多轮对话数据格式。  
例如：

```json
{
    "id": "abstract_v002_val2015_000000024714_000",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "scene_img_abstract_v002_val2017/abstract_v002_val2015_000000024714.png"
          },
          {
            "type": "text",
            "text": "Is the football being used right now?"
          }
        ]
      },
      {
        "role": "assistant",
        "content": "no"
      },
      {
        "role": "user",
        "content": "Is the grill on?"
      },
      {
        "role": "assistant",
        "content": "yes"
      }
    ],
    "source_ids": [
      "0247142",
      "0247140"
    ]
  },
```

转换示例：  
```python
def build_multiturn_samples(items, min_turns, max_turns, shuffle, seed, rewrite_repeated_questions, image_prefix):
    grouped = defaultdict(list)
    skipped = 0

    for item in items:
        qa = extract_first_qa(item)
        if qa is None:
            skipped += 1
            continue
        grouped[qa["image"]].append(qa)

    rng = random.Random(seed)
    output = []
    dropped_below_min = 0

    for image, qas in grouped.items():
        if shuffle:
            rng.shuffle(qas)

        for chunk_idx, start in enumerate(range(0, len(qas), max_turns)):
            chunk = qas[start:start + max_turns]
            if len(chunk) < min_turns:
                dropped_below_min += len(chunk)
                continue

            messages = []
            seen_questions = set()
            source_ids = []
            image_ref = join_image_ref(image_prefix, image)

            for turn_idx, qa in enumerate(chunk):
                question = qa["question"]
                if rewrite_repeated_questions:
                    question = maybe_rewrite_repeated_question(question, seen_questions, turn_idx)

                append_user_message(
                    messages,
                    question=question,
                    image_ref=image_ref if turn_idx == 0 else None,
                )
                append_assistant_message(messages, qa["answer"])
                if qa["id"]:
                    source_ids.append(qa["id"])

            sample_id = f"{os.path.splitext(os.path.basename(image))[0]}_{chunk_idx:03d}"
            output.append({
                "id": sample_id,
                "messages": messages,
                "source_ids": source_ids,
            })

    return output, {
        "input_samples": len(items),
        "image_groups": len(grouped),
        "output_samples": len(output),
        "skipped_invalid_samples": skipped,
        "dropped_turns_below_min_turns": dropped_below_min,
    }
```

格式转换的思路是利用每张图片的多个描述或多个 qa 进行多轮对话合并构造。  
第一轮对话会保留 **image**，用于后期替换成图片 embeding，后面轮次只保留文本，对于 coco 中的 caption 构造了几种不同的提问方式，随机进行提问。  

格式对齐多轮对话之后，将 dataset.py 中加载数据集 json 文件的部分修改成支持 OpenAI messages的格式，同时兼容 LLaVA 的 conversations 格式。  

```python
class LlavaPretrainDataset(Dataset):
    """读取图文对话数据的 Dataset。

    这个类现在同时支持旧 LLaVA conversations 和 OpenAI chat messages。
    每条样本只返回 PIL 图片和结构化 turns；
    tokenize、padding、labels mask 都在 LlavaCollator 里完成。
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
```
同时，不要忘记在 LlavaCollator 中将多轮对话第一轮保存 **image** 字段，同时屏蔽掉 pad 区域和 user 部分，将这几块区域的 label设置为 -100。  

```python
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
```

推理时，由于是多轮对话了，也要变化一下逻辑。从只能问一句，到能多轮追加提问。  
一种很直觉的方式是保存之前的对话内容，每次更新历史记录。  
在 infer.py 中通过 build_context_prompt() 函数更新历史，保证每次提问后将之前的全部对话和新问题同时送到模型中。  
我尝试在 prompt 中让模型以 json 的格式返回，但是失败了，模型并没有使用 json 格式。可能是因为微调数据集中没有 json 格式的分布，模型学到的多半是看图回答问题的任务模式，模仿训练数据里的回答风格。  

```python
def answer_one_turn(model, image, history, question, gen_config):
    """执行一轮带上下文的图文对话推理，并返回 assistant 回答。"""
    prompt = build_context_prompt(history, question)
    outputs = model.generate(
        images=[image],
        prompts=[prompt],
        max_new_tokens=int(gen_config["MAX_NEW_TOKENS"]),
        temperature=float(gen_config["TEMPERATURE"]),
        do_sample=bool(gen_config["DO_SAMPLE"]),
        top_p=float(gen_config["TOP_P"]),
        top_k=int(gen_config["TOP_K"]),
        repetition_penalty=float(gen_config["REPETITION_PENALTY"]),
    )
    return normalize_generated_text(outputs[0], prompt)
```

当然这种方式如果一直拼接的话会使得对话非常长，主流的解决方案有几种，在实际生产过程中比较常见：  

1、历史记录只保存第一轮和最近几轮对话  
2、将早期历史自动摘要  
3、多模态内容结构化存储  
4、向量检索相关历史  

例子中就简单的将所有历史对话全部保存了。  

初步验证：模型在 vqa 和 coco caption 的 val 集上微调了三个 epoch 后看下效果：  

```shell

python week11_multiturn_dialogue/code/infer.py   --image dataset/coco128/images/train2017/000000000025.jpg   --interactive   --context-file week11_multiturn_dialogue/outputs/context/demo_chat.json

```

```text

进入多轮对话。输入 exit/quit/q 结束，输入 clear 清空上下文。
USER: What color is the main object?
ASSISTANT: Green
USER: Please describe this image.
ASSISTANT: A tall giraffe standing in a green field.
USER: q

```

模型能够实现多轮对话了，但还有很多不完美。  

比如一直拼接，history 会越来越长，导致最后模型的输入 token 不够用。模型目前只能在第一轮对话中输入图片，后续轮对话中不能随时传文件，这和构建数据集的方式有关，我们的数据集是利用 caption 和 qa 强行改造的多轮对话数据集。实际生产过程中的数据里，其他模态数据理应是可以穿插在任何轮对话的任何位置中的。  

多轮对话的原理还是在数据和工程上做了些文章，本期并不涉及到模型方面的改动。至此 minillava 已经从简单的图像caption能力扩展到能同时训练多任务、多轮对话的多模态模型了，初见成效。  

在这个过程中也逐渐看清了多模态的大模型的具体流程。还有很多可优化的细节，我们下期继续。



