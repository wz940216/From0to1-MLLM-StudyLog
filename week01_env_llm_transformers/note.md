# 环境搭建 & Transformers 入门

## Transformer文档手册

https://hugging-face.cn/docs/transformers/index

## HF注册 获取token

信息填写时国家写美国，尽量填写全

## 模型下载

**下载hf**

```shell
curl -LsSf https://hf.co/cli/install.sh | bash
```

**登录**

```shell
hf auth login
```

**下载模型**

```shell
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Llama-3.1-8B-Instruct
hf download Salesforce/blip-image-captioning-base --local-dir models/blip-image-captioning-base
hf download Salesforce/blip2-flan-t5-xl --local-dir models/blip2-flan-t5-xl
hf download openai/clip-vit-base-patch16 --local-dir models/clip-vit-base-patch16
hf download llava-hf/llava-1.5-7b-hf --local-dir models/llava-1.5-7b-hf
hf download Qwen/Qwen1.5-1.8B-Chat --local-dir models/Qwen1.5-1.8B-Chat

```


## 调参

### max_new_tokens

中文大约 1 token ≈ 0.7–1 个汉字（跟分词器有关，只做粗略估算）。  
比如想要 800–1000 字的回答，可以设置 max_new_tokens ≈ 800 ~ 1200，再视显存和速度调整。

### top_p

把所有 token 按概率从大到小排序；  
从上往下累加概率，直到和 ≥ top_p；  
只在这个“核”里面采样，其余低概率词直接丢弃。

和 top_k 的区别：  
top_k：保留前 k 个最高概率词。  
top_p：保留概率累积到 p 的那批词，数量是动态的。

### temperature

把模型算出的 logits 除以一个系数 T，再做 softmax：  
T = temperature  
T < 1：把分布压得更“尖”  
T > 1：把分布拉得更“平”  

temperature 小（接近 0）  
把“原本就高概率”的词拉得更高  
把“原本就低概率”的词压得更低  
输出更稳定、重复性强、很少“冒险”  

temperature 大（>1）  
高概率词和中等概率词差距被缩小  
模型更愿意尝试非主流词  
输出更多样，有创意，但也更容易胡言乱语  

|        任务类型         |     temperature      |   top_p    |       说明       |
| :---------------------: | :------------------: | :--------: | :--------------: |
|   事实问答 / 检索问答   |      0.0 – 0.3       | 0.8 – 1.0  | 降低胡编乱造概率 |
|        文本翻译         |      0.0 – 0.3       | 0.9 – 1.0  |  尽量确定性输出  |
|   摘要 / 关键信息抽取   |      0.1 – 0.4       |    0.9     | 稍微留一点多样性 |
| 代码补全 / 单元测试生成 |      0.0 – 0.3       | 0.8 – 0.95 |    不要太随机    |
|     教学/解释型回答     |      0.4 – 0.7       |    0.9     |   回答自然一些   |
|        正常聊天         |      0.6 – 0.8       |    0.9     |    更像人聊天    |
|     文案/故事/脑暴      | 0.8 – 1.0 (最多 1.2) | 0.9 – 0.95 | 高多样性，高创意 |