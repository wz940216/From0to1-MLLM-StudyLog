# minillava_training_v1
上一篇搭建了minillava中的三大组件，vision encoder，projector，llmdecoder。这篇将持续完善训练pipline。  
包括dataset、trainloop、infer三个部分。  
实现完整的minillava的训练和推理。  

实验中我冻结了clip的模型参数，训练projector和lora的qwen1.5。  
在本没有多模态功能的qwen1.5上，经过3个epoch的微调，每个epoch只取LLaVA-CC3M的一千个样本，用时大概半小时。  
从实验结果看，模型能初步看懂图片和文字描述，开始具备多模态能力。  
但仍然比较垃圾😈。。

<div style="text-align: center;">
  <img src="../dataset/coco128/images/train2017/000000000025.jpg" alt="" style="zoom:50%;" />
</div>

**user**: "What is in the picture?"  
**minillava**: "pacific bluebird at a nest in the tree ."  
我问他图片里有什么？他回答说太平洋鸟在树旁边...嗯 至少树看懂了。  
相信如果在LLaVA-CC3M上完整训练3个epoch后模型的理解能力会更强。  

## dataset
数据集格式以LLaVA-CC3M的chat.json标注文件为例。
```json
[
  {
    "id": "GCC_train_002582585",
    "image": "GCC_train_002582585.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Provide a brief description of the given image.\n<image>"
      },
      {
        "from": "gpt",
        "value": "olive oil is a healthy ingredient used liberally ."
      }
    ]
  },
]
```
标注中包括每个图片名称和对话细节，其中对话角色分为human和gpt，防止角色错乱。  
value是prompt和llm要预测的句子。  
<image>是一个特殊字符，用于占位图片特征。  
在dataset中，我们需要解析出图片路径，以及对应的prompt和label。  
同时将图像特征和文字prompt结合起来组成多模态输入。  
组合的方式有很多种，可以直接将特殊字符处的text embedding替换为image embedding。  
也可以直接将特殊字符去掉，直接将图片embedding拼接到text embeding前面。  
这里我们采用第二种较为简单的方式，只要保证训练和推理时构建多模态prompt的方式一致即可。  

```python

def build_prompt(question):
    """构造训练和推理保持一致的文本模板。

    这里用简单清晰的 Q/A 模板，便于理解。真正大规模训练时也可以换成
    Qwen chat template，但要保证训练和推理使用同一套格式。
    """
    question = _clean_text(question)
    return f"User：{question}\nAssistant"

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
```
从json文件中解析出一张图片的标注后，通过_clean_text直接去掉prompt中的特殊字符。  
然后拼接处出我们自己的minillava多模态prompt。  
```python
f"User：{question}\nAssistant"
```

包装成dataset类  

```python
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
        for offset in range(len(self.samples)):
            sample_index = (index + offset) % len(self.samples)
            item = self.samples[sample_index]
            image_path = os.path.join(self.image_dir, item["image"])
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                # print(f"无法打开图片文件 {image_path}，将换一张图片。异常信息：{e}")
                continue

            question, answer = extract_qa(item["conversations"])
            return {
                "image": image,
                "prompt": build_prompt(question),
                "answer": answer,
                "image_path": image_path
            }

        raise RuntimeError("所有样本图片都无法打开，请检查图片目录和标注文件。")
```

到这里还没有结束，为了训练时方便，还需要一个函数将text prompt通过tokenizer处理成token id的形式  
并将label中的有关text prompt部分的token id以及padding部分的id设置为-100，进行屏蔽。  
因为我们需要llmdecoder在计算损失时忽略已知的prompt信息，只对生成的信息计算损失。
```python
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
```
到此dataset构建完毕。

## trainloop
在trainloop中，我们需要提前准备优化器、学习率调度器、模型保存模块。  
之后加载dataset，包装成dataloder之后就可以循环取数据开始训练了。  

**优化器部分**
```python
def build_optimizer(model, config):
    """根据配置创建优化器，只更新 requires_grad=True 的参数。"""
    # 训练配置中 OPTIMIZER 字段决定优化器类型、学习率和权重衰减等参数。
    optim_config = config["TRAINING"]["OPTIMIZER"]
    # 过滤掉被冻结的视觉编码器或语言模型参数，只训练当前允许更新的部分。
    params = [p for p in model.parameters() if p.requires_grad]
    optim_type = optim_config["TYPE"].lower()
    # AdamW 是大语言模型微调中最常用的优化器，带 decoupled weight decay。
    if optim_type == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(optim_config["LR"]),
            weight_decay=float(optim_config["WEIGHT_DECAY"]),
            betas=tuple(optim_config["BETAS"])
        )
    # Adam 不使用 AdamW 的解耦权重衰减，适合简单调试或对比实验。
    if optim_type == "adam":
        return torch.optim.Adam(params, lr=float(optim_config["LR"]))
    # SGD 一般不用于 LLM 微调，但保留入口便于实验。
    if optim_type == "sgd":
        return torch.optim.SGD(params, lr=float(optim_config["LR"]), momentum=0.9)
    raise ValueError(f"不支持的优化器类型: {optim_type}")
```
**学习率调度**  
```python
def build_scheduler(optimizer, config, total_steps):
    """创建学习率调度器。"""
    # 调度器配置决定 warmup 步数、总训练步数下学习率如何变化。
    sched_config = config["TRAINING"]["SCHEDULER"]
    warmup_steps = int(sched_config["WARMUP_STEPS"])
    sched_type = sched_config["TYPE"].lower()
    # cosine：warmup 后按余弦曲线逐渐衰减，常用于 Transformer 训练。
    if sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # linear：warmup 后线性衰减到 0，行为更直观。
    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # constant：不使用 scheduler，训练过程中学习率保持 optimizer 初始值。
    if sched_type == "constant":
        return None
    raise ValueError(f"不支持的调度器类型: {sched_type}")
```
**模型保存**  
保存模型时可以选择只保存我们开放训练的参数，来提高保存速度，减少硬盘消耗。  
accelerator prepare后的model可能被DDP/FSDP等包装，保存前要取回原始模型对象。  
```python
def save_checkpoint(accelerator, model, optimizer, scheduler, step, save_dir):
    """保存训练检查点，包含 projector、可训练参数和优化器状态。"""
    # 多卡训练时每个进程都会执行代码；只让主进程写文件，避免多个进程同时覆盖同一路径。
    if not accelerator.is_main_process:
        return

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
    # prepare 后的 model 可能被 DDP/FSDP 等包装；保存前要取回原始模型对象。
    unwrapped_model = accelerator.unwrap_model(model)
    # 只保存开启梯度的参数和 LoRA adapter 参数，冻结的基础模型权重由初始化模型提供。
    saved_param_names = [
        name
        for name, param in unwrapped_model.named_parameters()
        if should_save_param(name, param)
    ]
    saved_state_dict = {
        name: param.detach().cpu()
        for name, param in unwrapped_model.named_parameters()
        if should_save_param(name, param)
    }
    # accelerator.save 会在分布式环境中安全保存对象，语义类似 torch.save。
    accelerator.save(
        {
            # 记录当前全局步数，方便后续恢复或排查 checkpoint 来源。
            "step": step,
            # 只保存部分参数；加载时用参数名称匹配，并允许未保存的冻结参数缺失。
            "model": saved_state_dict,
            # 显式记录保存了哪些参数，便于检查 checkpoint 内容。
            "saved_param_names": saved_param_names,
            # 保存优化器状态，恢复训练时可以延续动量等内部统计量。
            "optimizer": optimizer.state_dict(),
            # constant scheduler 为 None，其它 scheduler 保存状态用于恢复学习率进度。
            "scheduler": scheduler.state_dict() if scheduler is not None else None
        },
        ckpt_path
    )
    print(f"已保存检查点: {ckpt_path}")
```

训练时，加载了hf的accelerator库，accelerator是一个很便捷的分布式训练管理包  
accelerator可以接管混合精度训练、梯度裁剪、反向传播、梯度累计等常用功能  
通过accelerator可以轻松的实现大餐数量模型在多卡上的分片训练  
```python
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```
使用时只需要用accelerator包装好模型、优化器和数据迭代器即可  

另外scheduler的总步数应该按accelerate分片后的dataloader长度来算，否则多卡时学习率会按单卡步数走得偏慢。  
把模型、优化器、dataloader、scheduler 交给 Accelerator.prepare()，反向传播、梯度裁剪、日志和保存都走accelerate的多进程安全接口。  
每个进程拿自己的dataloader shard，反向传播和梯度裁剪用accelerate接管，checkpoint只由主进程写。    

**完整train loop**  
```python
def main():
    # 命令行参数只保留配置路径和调试样本数，其他训练参数统一放在 YAML 中管理。
    parser = argparse.ArgumentParser(description="MiniLLaVA 微调脚本")
    parser.add_argument("--config", default="week08_minillava_training_v1/code/config.yaml", help="训练配置文件路径")
    parser.add_argument("--max-samples", type=int, default=1000, help="调试时只取前 N 条数据")
    args = parser.parse_args()

    # Accelerator 会根据 accelerate launch 的启动方式自动识别进程数、设备和分布式后端。
    accelerator = Accelerator()
    config = load_config(args.config)
    # 保留原本的 PyTorch 随机种子设置。
    set_seed(int(config["MISC"]["SEED"]))
    # 再使用 Accelerate 的种子工具，保证多进程场景下每个进程的随机状态可控。
    accelerate_set_seed(int(config["MISC"]["SEED"]))

    # 初始化 MiniLLaVA，并切换到训练模式，启用 dropout 等训练期行为。
    model = MiniLlavaModel(args.config)
    model.train()

    # 从配置中读取训练数据路径、图片目录、标注文件、batch size 等数据相关参数。
    train_config = config["DATA"]["TRAIN_DATASET"]
    dataset = LlavaPretrainDataset(
        dataset_path=train_config["PATH"],
        image_dir=train_config["IMAGE_DIR"],
        annotation_file=train_config["ANNOTATION_FILE"],
        max_samples=args.max_samples
    )
    # Collator 在 DataLoader 拼 batch 时进行 tokenizer、padding，并构造只监督 answer 的 labels。
    collator = LlavaCollator(
        tokenizer=model.language_decoder.tokenizer,
        max_length=int(train_config["MAX_LENGTH"])
    )
    # DataLoader 仍按普通 PyTorch 写法创建；后面 accelerator.prepare 会自动处理多卡分片。
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config["BATCH_SIZE"]),
        shuffle=True,
        num_workers=int(train_config["NUM_WORKERS"]),
        collate_fn=collator,
        pin_memory=torch.cuda.is_available()
    )

    # 先在原始 model 上创建优化器，这样 optimizer 能拿到正确的可训练参数列表。
    optimizer = build_optimizer(model, config)
    num_epochs = int(config["TRAINING"]["SCHEDULER"]["NUM_EPOCHS"])
    # prepare 会把 model 放到正确设备，并在多卡时包装为分布式模型；
    # dataloader 也会被切成每个进程各自负责的一份数据。
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # prepare 之后的 dataloader 长度是当前进程实际迭代步数，用它计算 scheduler 总步数更适合多卡。
    total_steps = len(dataloader) * num_epochs
    scheduler = build_scheduler(optimizer, config, total_steps)
    # scheduler 依赖已经 prepare 过的 optimizer，因此在 optimizer prepare 之后创建并交给 accelerator。
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    # 日志、保存、梯度裁剪等训练控制参数。
    log_steps = int(config["TRAINING"]["LOGGING"]["LOG_STEPS"])
    save_steps = int(config["TRAINING"]["CHECKPOINT"]["SAVE_STEPS"])
    save_dir = config["TRAINING"]["CHECKPOINT"]["SAVE_DIR"]
    max_norm = float(config["TRAINING"]["GRAD_CLIP"]["MAX_NORM"])

    # global_step 记录当前进程执行的优化步数；多卡下各进程同步前进。
    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            # batch 来自 LlavaCollator：
            # images 是 PIL 图片列表，input_ids/attention_mask/labels 是已 padding 的张量。
            outputs = model(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            # MiniLlavaModel 最终调用语言模型，labels 存在时 transformers 输出中会包含 loss。
            loss = outputs.loss
            # 使用 Accelerator 进行反向传播，兼容多卡、混合精度和梯度累积等能力。
            accelerator.backward(loss)

            # 梯度裁剪可以缓解训练初期或小 batch 时的梯度爆炸。
            accelerator.clip_grad_norm_(model.parameters(), max_norm)
            # 参数更新。
            optimizer.step()
            # 如果启用了 scheduler，每个优化步后推进一次学习率。
            if scheduler is not None:
                scheduler.step()
            # set_to_none=True 可以减少显存写入，下一次 backward 时再重新分配梯度。
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            # 聚合所有进程上的 loss，日志展示的是多卡平均 loss，而不是单个进程的局部 loss。
            loss_value = accelerator.gather_for_metrics(loss.detach()).mean().item()
            
            # 只让全局主进程打印日志，避免多进程重复输出相同 step。
            if global_step % log_steps == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch + 1}/{num_epochs} step={global_step} loss={loss_value:.4f} lr={lr:.8f}")
                
            # 到达保存间隔时，先等待所有进程到同一步，再由主进程写 checkpoint。
            if global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)

    # 训练结束后再同步一次，确保所有进程都完成最后一个 epoch。
    accelerator.wait_for_everyone()
    # 保存最终 checkpoint；函数内部会判断是否为主进程。
    save_checkpoint(accelerator, model, optimizer, scheduler, global_step, save_dir)
```

模型训练运行方式示例：
```shell
accelerate launch --num_processes 2 week08_minillava_training_v1/code/train.py
```
或者先配置：
```shell
accelerate config
accelerate launch week08_minillava_training_v1/code/train.py
```

## infer
推理部分相对简单，加载我们训练时保存的模型，加载图片，构建prompt后调用generate进行推理即可。
```python
def main():
    parser = argparse.ArgumentParser(description="MiniLLaVA推理脚本")
    parser.add_argument("--config", default="week08_minillava_training_v1/code/config.yaml")
    parser.add_argument("--checkpoint", default="week08_minillava_training_v1/outputs/checkpoints/step_3000.pt", help="训练得到的 .pt 检查点路径")
    parser.add_argument("--image", default="dataset/coco128/images/train2017/000000000009.jpg", help="输入图片路径")
    parser.add_argument("--question", default="description this picture.", help="关于图片的问题")
    args = parser.parse_args()

    model = MiniLlavaModel(args.config)
    if args.checkpoint is not None:
        # 检查点只保存可训练参数和 LoRA adapter；需用同样 LoRA 配置初始化模型后按名称加载。
        state = torch.load(args.checkpoint, map_location=model.device)
        model.load_state_dict(state["model"], strict=False)

    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.question)
    gen_config = model.config["INFERENCE"]["GENERATION"]
    outputs = model.generate(
        images=[image],
        prompts=[prompt],
        max_new_tokens=int(gen_config["MAX_NEW_TOKENS"]),
        temperature=float(gen_config["TEMPERATURE"]),
        do_sample=bool(gen_config["DO_SAMPLE"]),
        top_p=float(gen_config["TOP_P"]),
        top_k=int(gen_config["TOP_K"]),
        repetition_penalty=float(gen_config["REPETITION_PENALTY"])
    )
    print(outputs[0])
```
## 总结
本周实现了自己搭建mini版llava模型。  
实现了accelerator多卡分布式训练。  
实现了多模态数据集加载和dataloder的构建。  
实现了多模态数据训练所需的label、attention_mask的构建。  
完成了第一个版本的minillava模型。  
后续会基于初版minillava进一步进行探索。  
实现模块化重构、单元测试、多任务训练、多轮对话支持、指令对齐、安全策略、模型部署等工作。

