# week17_llm_inference_vllm

vLLM 是一个 LLM 推理/服务引擎。它跑的仍然是 Transformer 模型，比如 Llama、Qwen、Mistral，只是把怎么调度请求、怎么管理 KV cache、怎么高效占满 GPU这件事做得比普通 transformers.generate() 更像生产级服务系统。举个例子：Transformers 单独推理像是你自己开车一趟一趟送人；vLLM 像调度中心，把很多乘客动态拼车，并且把车内座位管理得非常细。是一个集成了显存调度、管理、模型量化、算子融合、多线程/多进程、异步请求等技术的推理引擎。vLLM 也支持 LoRA、PEFT、LoCon 等 adapter 推理。

## 普通 Transformers 推理在做什么

以 Hugging Face transformers 为例，典型流程是：
```text

model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)
output = model.generate(...)

```

模型生成 token 时，本质是自回归：
先处理 prompt，计算第一步所需的 hidden states。每生成一个 token，就把这个 token 再喂回模型。  
为了避免每一步都重算历史 token 的 Key/Value，会维护 KV cache。下一步 attention 只计算当前 token 的 query，但会读取历史所有 token 的 cached K/V。  
这对单请求、实验、debug 很方便。但如果同时服务很多用户，问题就来了：每个请求长度不同、生成长度未知、KV cache 动态增长，GPU 内存会被浪费得很厉害。  

## vLLM 主要解决什么问题

vLLM 解决的核心是多个请求怎么高效批处理，每个请求的 KV cache 怎么放进 GPU 内存？请求来了、结束了、变长了，怎么动态调度？怎么让 GPU 尽量别空着？长上下文、多并发、多采样、beam search 怎么减少内存浪费？  

beam search 在这里是第一次出现。官方的说 beam search 是一种生成策略，通常用于生成更高质量的文本。它会在每一步生成多个候选 token（称为 beams），并根据概率选择最优的几个 beams 继续生成。这样可以探索更多的可能性，但也会增加 KV cache 的需求，因为每个 beam 都需要维护自己的缓存。  
但其实 beam search 就是在生成 token 中保留最优的 beam size 个候选序列，并且每一步只维护 beam size 个最好的结果。例如：

```text
beam size = 3
我 喜欢 吃 ...
我 喜欢 看 ...
我 喜欢 学习 ...
```

beam search 每一步会比较这些候选序列的分数，保留分数最高的 beam size 条。  
不过有个细节：只用概率相乘会偏爱短句。因为每多生成一个 token，就要再乘一个小于 1 的概率，整体概率通常会下降。所以实际 beam search 经常会加一个 长度惩罚 / 长度归一化：  
score = log概率总和 / 长度惩罚项  
这样不会让模型过分倾向特别短的输出。  
具体算的时候，通常是累计每个 token 的 log probability，再配合长度惩罚等修正。  

vLLM 论文[Efficient Memory Management for Large Language
Model Serving with PagedAttention](../docs/notes/vllm.pdf)指出，LLM serving 的瓶颈很大程度来自 KV cache。模型权重常驻显存，activation 只是短暂存在，而 KV cache 会随请求数和序列长度增长，管理不好会限制 batch size 和吞吐。  

## vllm核心原理：PagedAttention

vLLM 最核心的思想是 PagedAttention。普通做法容易把一个请求的 KV cache 当作一整段连续内存来管理。但问题是：输出长度事先不知道。输入有的请求很短，有的很长。如果预留最大长度，会浪费显存。如果动态申请连续显存，会导致显存碎片化，找不到连续大块的显存可用，有涉及到显存搬移。有时多个采样分支、beam search 中有大量共享前缀，但普通连续 cache 不好共享。  

PagedAttention 借鉴操作系统的虚拟内存分页思想，将显存分成了不同个 block 单元。每个请求的 KV cache 不再要求连续，而是按需分配到不同 block 中。程序上看起来还是一个连续的 token 序列，但物理显存中可以分散存放在固定大小的 KV block 里。  

```text
PagedAttention：
request A logical blocks:
  block 0 -> physical block 17
  block 1 -> physical block 03
  block 2 -> physical block 91
```

也就是说，一个请求逻辑上还是连续的 token 序列，但物理显存中可以分散存放在固定大小的 KV block 里。  

这种分配的好处是：  
- 按需分配：生成到哪里，KV block 分配到哪里，动态申请动态释放。  
- 减少内部碎片：不用为最大输出长度提前占满，没占满一个 block 后分配下一个，也就最后一块 block 可能出现内部碎片。  
- 减少外部碎片：所有 block 大小一致，更容易复用，外部不需要一块很大的且连续的存储块。  
- 支持共享：共享 prompt、beam search、多采样时，多个序列可以引用同一批 KV block。  
c- opy-on-write：分支需要改写时再将需要改的 block 复制一份，改复制的那一份即可，其他 block 不需要改变，仍可复用。

## continuous batching

普通 batching 常见做法是：收集一批请求，一起跑完。问题是 LLM 生成长度不同：

```text
request A: 生成 10 token
request B: 生成 300 token
request C: 生成 50 token
```

如果等整批结束，短请求会被长请求拖住；如果 padding 到一样长，计算又浪费。  

vLLM 使用更细粒度的调度，通常称为 continuous batching / iteration-level scheduling：  
做法是这样的，假如有四个 batch 请求A/B/C/D：  

step 1: A B C D 一起生成   [A B C D]  
step 2: A 完成后将 E 加入   [A B C D] -> [B C D E]   
step 3: C 完成，F 加入     [B C D E] -> [B D E F]   
step 4: B D E F 继续生成    

也就是每个 decoding step 都可以重新组织 batch。vLLM 架构中 engine core 会运行 scheduler，管理 KV cache，并协调 GPU worker 执行模型。  

## vLLM 的优点

1、吞吐高：多并发下优势明显。原论文报告在评估场景中相对 FasterTransformer、Orca 达到 2-4 倍吞吐提升，尤其长序列、大模型、复杂 decoding 更明显。  
2、显存利用率高：PagedAttention 减少 KV cache 浪费。  
3、适合在线服务：支持 OpenAI-compatible API server。  
4、调度能力强：continuous batching、preemption、chunked prefill 等。  
5、长上下文更友好：KV cache 是长上下文推理的主要压力点之一，vLLM 正是围绕它优化。  
6、生产能力多：tensor parallel、pipeline parallel、quantization、LoRA、prefix caching、speculative decoding、metrics 等。  
7、共享前缀省显存：相同 prompt、多采样、beam search 等场景收益明显。  

## vLLM 的缺点
1、不是所有模型都支持：Transformers 通常是模型支持最全的地方；vLLM 对模型结构、attention 实现、tokenizer、特殊 forward 逻辑有要求。  
2、部署复杂度更高：服务进程、GPU worker、scheduler、显存参数、并行参数都要调。  
3、小并发未必划算：如果只是本地单条 prompt 测试，Transformers 更轻、更直观。  
4、调参敏感：max_num_batched_tokens、max_num_seqs、gpu_memory_utilization、并行策略会影响 TTFT、ITL、吞吐、OOM 风险。  
5、调试模型内部更麻烦：研究时想改 attention、插 hook、看 hidden states，Transformers 更顺手。  
6、预占显存明显：vLLM 通常会预留较多显存给 KV cache。  
极端情况下会 preempt/recompute，也就是当显存不够时，KV cache 空间不足会抢占请求，为了不 OOM, 先暂停某些请求并释放它们的 KV cache，后续再重算，这能保证系统健壮性，但会影响端到端延迟。为什么 vllm 不将 KV cache 先临时放到 CPU 内存，后续再搬回 GPU 呢，猜测也许是 KV catch 很大，在没有 NVLink 的情况下，搬运数据也比较占资源。  

## 什么时候用哪个
用 Transformers 直接推理：
- 研究、实验、调模型结构。
- 单用户、低并发。
- 模型很新，vLLM 暂不支持。
- 需要完全控制 forward、logits、hidden states。
- 本地简单 demo。  

用 vLLM：
- 要部署 API 服务。
- 多用户并发。
- 需要高吞吐、低成本。
- prompt/output 长度差异大。
- 长上下文、多采样、beam search。
- 想要 OpenAI-compatible server。
- 要做生产监控、并行、量化、LoRA serving。  

## vllm 应用
vllm 的调用方式和 transformers 很像，都是通过 `generate()` 来生成 token。

一个简单的例子：  

```python

def benchmark_vllm(args: argparse.Namespace) -> BenchResult:
    from vllm import LLM, SamplingParams
    # 加载原始模型的 tokenizer，vllm 也有自己的 tokenizer，但为了和 transformers 保持一致，这里直接用原始模型的 tokenizer
    tokenizer = load_tokenizer(args.model_path)
    prompt = build_chat_prompt(tokenizer, args.prompt)
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p,
    )
    # vllm 的 LLM 类是一个高性能的推理引擎，支持多并发、显存管理、量化等特性
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.concurrency,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.warmup > 0:
        llm.generate(
            [prompt] * args.warmup,
            sampling_params,
            use_tqdm=False,
        )
    prompts = [prompt] * args.num_requests
    start = time.perf_counter()
    # generate() 方法会返回一个列表，每个元素是一个生成结果对象，包含生成的 token ids、logits 等信息
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start

    output_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
    return BenchResult(
        backend="vllm",
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        elapsed_sec=elapsed,
        qps=args.num_requests / elapsed,
        output_tokens=output_tokens,
        tokens_per_sec=output_tokens / elapsed,
    )

```

也可以起一个 server ：

```python

def run_vllm_server(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    print("Starting vLLM server:")
    print(" ".join(command))
    subprocess.run(command, check=True)

```
