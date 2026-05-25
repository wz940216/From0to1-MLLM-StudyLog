# Multitask Data
第十周，开始让模型支持多任务数据，整体思路是先将所有数据格式转换成统一 llava conversation 格式，在按特定比例抽样，让模型同时训练多任务数据。  

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

我们再来看一下 llava 的数据格式：  

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
## 构建多任务数据集
将各个任务的数据集整理成统一格式后，如果直接放在 dataset 中加载，会产生数据集数量多的采样概率大，数据及数量小的，被采样的概率变小的问题。  
最好在加载数据集时，人为设置采样权重，平衡各个数据集的采样率。  

在配置文件里设置各个数据集任务的数据地址和采样率：  
```yaml
DATA:
  TRAIN_DATASET:
    TASK_NAME:
     - "LLaVA-CC3M-Pretrain"
     - "COCOCaption"
     - "VQA"
    PATH: 
     - "dataset/LLaVA-CC3M-Pretrain-595K"
     - "dataset/COCOCaption"
     - "dataset/VQA"
    IMAGE_DIR: 
     - "images"
     - "val2017"
     - "scene_img_abstract_v002_val2017"
    ANNOTATION_FILE: 
     - "chat.json"
     - "annotations/captions_val2017_qa.json"
     - "abstract_v002_val2017_qa.json"
    SAMPLE_RATE: 
     - 0.4
     - 0.3
     - 0.3
```
这里每个 list 的同一位置对应一个任务：

```text
第 0 个任务:
  name = LLaVA-CC3M-Pretrain
  path = dataset/LLaVA-CC3M-Pretrain-595K
  image_dir = images
  annotation = chat.json
  sample_rate = 0.4

第 1 个任务:
  name = COCOCaption
  path = dataset/COCOCaption
  image_dir = val2017
  annotation = annotations/captions_val2017_qa.json
  sample_rate = 0.3

第 2 个任务:
  name = VQA
  path = dataset/VQA
  image_dir = scene_img_abstract_v002_val2017
  annotation = abstract_v002_val2017_qa.json
  sample_rate = 0.3
```

train.py 里的 build_train_dataset() 会把这些 list 拆开，逐个创建 LlavaPretrainDataset。单个数据源仍然还是由LlavaPretrainDataset 来处理，我们只需要合并它们，并且给予对应的采样权重即可。  
合并方法： 
```python
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
```

使用 PyTorch 中的工具函数拼接索引空间
```python
ConcatDataset(self.datasets)
```
生成每个样本的权重  
一个很直觉的算法就是：  
某个任务中每条样本的权重 = 该任务采样比例 / 所有采样比例之和 / 该任务样本数  

例如 COCO 单条样本的权重会比 LLaVA 单条样本大很多，因为 COCO 样本少，但整个 COCO 任务仍然按照配置文件设置的，只占 30% 的抽样概率。  

采样时，使用 PyTorch 的工具函数，传入事先准备的 sample_weights ，并设置可放回采样 replacement=True ,表示同一个样本在一个 epoch 里可能被抽到多次，也可能一次都没抽到。
```python
WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)
```

来看一下整体流程：

```text
Caption / VQA / LLaVA chat
        ↓
统一成 conversations
        ↓
LlavaPretrainDataset
        ↓
MultiTaskLlavaDataset + ConcatDataset
        ↓
按 SAMPLE_RATE 生成逐样本权重
        ↓
WeightedRandomSampler 有放回采样
        ↓
LlavaCollator 构造 input_ids / labels
        ↓
MiniLLaVA 训练
```

多任务对话完成后，我们可以用这种方法联合训练各种开源数据集，下周计划再增加多轮对话能力，构建多轮对话数据集。