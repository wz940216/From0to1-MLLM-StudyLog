# Multitask Data

## COCO caption 和 QA 格式： 

caption 是**看图生成一句描述**，QA 指令是**看图后按问题回答**。  

**COCO Caption 格式**
```json
{
  "image": "xxx.jpg",
  "caption": "A dog is running on the grass."
}
```

训练目标通常是：

```text
输入：图像
输出：A dog is running on the grass.
```

特点：
- 没有明确问题
- 输出是完整自然语言描述
- 偏向图像整体内容描述
- 一张图通常有多条 caption
- 适合训练图像描述能力

**QA 指令格式**
```json
{
  "image": "xxx.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is the dog doing?"
    },
    {
      "from": "gpt",
      "value": "The dog is running on the grass."
    }
  ]
}
```

训练目标通常是：

```text
输入：图像 + 问题
输出：The dog is running on the grass.
```

特点：
- 有明确 instruction/question
- 输出针对问题，不一定描述整张图
- 可以是问答、判断、计数、推理、定位等
- 更接近多模态 Chat / Instruction Tuning
- 适合训练模型按用户指令回答

我们要把 COCO caption 转成 QA 指令格式，只需要在 caption中 加一个固定问题即可：

```json
{
  "image": "xxx.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nDescribe this image."
    },
    {
      "from": "gpt",
      "value": "A dog is running on the grass."
    }
  ]
}
```

也可以用中文：

```json
{
  "from": "human",
  "value": "<image>\n请描述这张图片。"
}
```

本质上，caption 数据可以看成一种特殊的 QA 数据：问题固定为请描述图片，答案就是 caption。

我们再来看一下llava的数据格式：  

**单轮对话**  
```json
{
  "id": "0000001",
  "image": "0000001.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n请描述这张图片。"
    },
    {
      "from": "gpt",
      "value": "图片中有一只橘猫趴在沙发上。"
    }
  ]
}
```  

**多轮对话**  
```json
{
  "id": "2",
  "image": "dog.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n图里有什么？"
    },
    {
      "from": "gpt",
      "value": "一只狗。"
    },
    {
      "from": "human",
      "value": "它在做什么？"
    },
    {
      "from": "gpt",
      "value": "它正在草地上奔跑。"
    }
  ]
}
```

## VQA（Visual Question Answering，视觉问答）

给定一张图片和一个问题，模型需要根据图片回答问题。  
VQA中一般有两个json文件，一个用来存储问题，一个用来存储答案。  

### 答案 annotations 字段含义：  

```json
{
  "question_type": "is it",
  "multiple_choice_answer": "yes",
  "answers": [...],
  "image_id": 28940,
  "answer_type": 'yes/no',
  "question_id": 289402
}
```

**image_id:** 28940  
对应图片的 ID。数据集中会有一张图片编号为 28940。  

**question_id:** 289402  
这个问题的唯一 ID。通常一个图片可以对应多个问题。  

**question_type:**'is it'  
问题类型，表示这个问题大概率是以 “Is it ...?” 开头，比如：  
Is it raining?  
Is it a dog?  
  
**answer_type:** 'yes/no'  
答案类型是“是/否”问题。  
  
**multiple_choice_answer:** 'yes'  
官方整理后的标准答案是 yes。  

**answers:** [{...}, {...}, ...]  
多个人类标注者给出的答案。VQA 通常会让 10 个人回答同一个问题，所以这里看到很多 {...}。里面一般类似：

```json
{
  "answer": "yes",
  "answer_confidence": "yes",
  "answer_id": 1
}
```

也就是说，不同标注者可能都回答了 yes，也可能有人回答 no 或其他近似答案。

### 问题 questions 字段含义：

```json
{
  "question_id": 289402,
  "image_id": 28940,
  "question": "Is it ...?"
}
```
**question_id** :对应 annotations 中的 question_id。  
**question**: 对应问题  

可以通过脚本查看，感受一下vqa数据集的格式：

```python 
import json

ann_path = "dataset/VQA/abstract_v002_val2017_annotations.json"
ques_path = "dataset/VQA/OpenEnded_abstract_v002_val2017_questions.json"

with open(ann_path, "r") as f:
    anns = json.load(f)["annotations"]

with open(ques_path, "r") as f:
    ques = json.load(f)["questions"]

qid = 289402

ann = next(x for x in anns if x["question_id"] == qid)
q = next(x for x in ques if x["question_id"] == qid)

print("image_id:", ann["image_id"])
print("question:", q["question"])
print("answer:", ann["multiple_choice_answer"])
print("all answers:", [a["answer"] for a in ann["answers"]])
```
