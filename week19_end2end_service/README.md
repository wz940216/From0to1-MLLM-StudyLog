# week19_end2end_service

第十九周了，这周在第 18 周 `week18_minillava_hf_vllm` 的 vLLM OpenAI-compatible 服务之上，封装一个端到端多模态 HTTP 服务。

这里的 FastAPI 不加载模型，它只是业务层 API；真正的模型推理在 vLLM 服务里完成。FastAPI 的价值是把上传图片、prompt 构造、限流、排队、日志、错误处理这些工程化逻辑包起来。

实现的流程大致是这样的：

```text
用户 / 客户端
  -> POST http://127.0.0.1:9000/chat
  -> FastAPI 接收 multipart 图片 + question
  -> FastAPI 做限流、排队
  -> 读取图片，PIL 校验/缩放，转成 base64 data URL
  -> 用图片原始 bytes 计算 SHA256，作为 vLLM image uuid
  -> 构造 OpenAI 多模态 messages
  -> httpx POST 到 http://127.0.0.1:8000/v1/chat/completions
  -> vLLM 加载 MiniLLaVA 模型做推理
  -> vLLM 返回 choices[0].message.content
  -> FastAPI 把 answer、图片信息、各阶段耗时返回给用户
```

这样拆成 FastAPI 业务服务 + vLLM 推理服务，核心好处是职责分离。  

vLLM 负责高效加载模型、管理 GPU、做 batch、KV cache、调度推理。FastAPI 负责处理业务逻辑：鉴权、上传图片、参数校验、限流、日志、错误处理、接口格式适配。两边互不掺杂，后面更容易维护。更符合工程化定义。  

比如：  

1、模型服务可以独立扩缩容

如果请求量变大，可以多起几个 vLLM 实例：
```text
FastAPI
  -> vLLM:8000
  -> vLLM:8001
  -> vLLM:8002
```
FastAPI 做负载均衡或路由，不需要改模型代码。

2、业务和模型解耦
比如以后加用户登录、权限校验、图像大小过滤、敏感词过滤等业务逻辑，可以放在 FastAPI。vLLM 不需要关心。

3、模型崩了不一定拖垮业务层

vLLM 因为 CUDA OOM、模型报错、GPU 重启挂了，FastAPI 还能返回清晰错误：
```text
{
  "detail": "vLLM is unavailable"
}
```
保持整个业务 API 可用。

4、方便替换后端模型  

只要接口不变，vllm 后端可以自由更换其他多模态大模型。业务不受影响。


## 功能

- `POST /chat`：上传图片和问题，返回 MiniLLaVA 回答
- 图片预处理：校验文件、RGB 转换、按最长边缩放、JPEG data URL 编码
- prompt 构造：使用 OpenAI 多模态消息格式，把图片和文本交给 vLLM
- 请求队列：用 `asyncio.Semaphore` 控制并发，避免服务被瞬时请求打满
- 简单限流：按客户端 IP 做固定窗口 QPM 限制
- 日志：记录预处理、prompt 构造、生成和总耗时
- 客户端脚本：`scripts/client.py`

队列作用是：限制同时进入模型推理流程的请求数量，防止 FastAPI 服务一下子把太多请求转发给 vLLM/GPU，导致显存、队列或延迟失控。  
在代码里它封装在 RequestQueue 中：  

```python
self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
/chat 请求会这样使用：
async with app.state.queue.slot():
    ...
    answer = await app.state.backend.generate(...)
```

如果 MINILLAVA_MAX_CONCURRENCY=2，同一时刻最多只有 2 个请求能进入图片预处理、prompt 构造和 vLLM 推理这段逻辑。  
第 3 个及之后的请求会等待 semaphore 释放名额。  
如果等待超过 MINILLAVA_QUEUE_TIMEOUT_S，会返回 503 request queue timeout。  
推理结束或发生异常后，finally 里会 release()，确保名额归还。  

限流的作用是限制每个 IP 每分钟最多多少次请求    
Semaphore 是整个服务同一时刻最多处理多少个重推理请求。  

## 启动

先启动第 18 周导出的 vLLM 服务：

```shell
conda run -n vllm_test python week18_minillava_hf_vllm/scripts/vllm_openai_server.py \
  --model-path week18_minillava_hf_vllm/outputs/vllm/minillava-hf \
  --served-model-name minillava \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --gpu-memory-utilization 0.85
```

再启动第 19 周业务服务：

```shell
conda run -n mllm uvicorn week19_end2end_service.service.app:app \
  --host 0.0.0.0 \
  --port 9000
```

调用接口：

```shell
conda run -n mllm python week19_end2end_service/scripts/client.py \
  --url http://127.0.0.1:9000/chat \
  --image week03_mllm_overview_llava_demo/code/image.png \
  --question "请描述这张图片。"
```

## 配置

通过环境变量调整：

- `MINILLAVA_VLLM_BASE_URL`：vLLM 服务地址，默认 `http://127.0.0.1:8000`
- `MINILLAVA_MODEL`：vLLM served model name，默认 `minillava`
- `MINILLAVA_MAX_CONCURRENCY`：业务服务并发数，默认 `2`
- `MINILLAVA_QUEUE_TIMEOUT_S`：排队超时时间，默认 `30`
- `MINILLAVA_RATE_LIMIT_PER_MIN`：单 IP 每分钟请求数，默认 `30`
- `MINILLAVA_MAX_IMAGE_MB`：上传图片大小上限，默认 `8`
- `MINILLAVA_MAX_IMAGE_SIDE`：图片最长边缩放上限，默认 `1024`
- `MINILLAVA_DEFAULT_MAX_TOKENS`：默认生成长度，默认 `128`
- `MINILLAVA_DEFAULT_TEMPERATURE`：默认采样温度，默认 `0.0`

## 测试

不依赖真实 vLLM 的冒烟测试：

```shell
conda run -n mllm python week19_end2end_service/scripts/smoke_test.py
```

单元测试：

```shell
conda run -n mllm pytest week19_end2end_service/tests
```

