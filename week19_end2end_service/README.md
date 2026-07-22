# week19_end2end_service

第十九周任务：在第 18 周 `week18_minillava_hf_vllm` 的 vLLM OpenAI-compatible 服务之上，封装一个端到端多模态 HTTP 服务。

## 功能

- `POST /chat`：上传图片和问题，返回 MiniLLaVA 回答
- 图片预处理：校验文件、RGB 转换、按最长边缩放、JPEG data URL 编码
- prompt 构造：使用 OpenAI 多模态消息格式，把图片和文本交给 vLLM
- 请求队列：用 `asyncio.Semaphore` 控制并发，避免服务被瞬时请求打满
- 简单限流：按客户端 IP 做固定窗口 QPM 限制
- 日志：记录预处理、prompt 构造、生成和总耗时
- 客户端脚本：`scripts/client.py`

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

