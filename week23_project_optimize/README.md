# week23_project_optimize

第 23 周 Beta：在 `week22_project_core_impl` Alpha 基础上做优化和工程加固。主链路仍然是 PDF 拆页、基础 OCR JSON、OCR 文本 + 相关页图片送入本地 Qwen3-VL 问答。

## Beta 优化点

- 多页耗时优化：新增分阶段 timings，记录 hash、cache load、split、OCR、VLM generate 等耗时。
- Prompt 优化：支持 `auto/general/table/heading/summary`，默认中文回答，并要求关键结论带页码依据。
- 文档缓存：默认按 PDF SHA256 缓存拆页、OCR JSON、artifacts，重复文档跳过拆页和 OCR。
- 答案缓存：默认关闭，可通过参数或环境变量开启，按文档 hash、问题、prompt、max_images 等生成 cache key。
- OCR 并发参数：`DOC_QA_OCR_WORKERS` 已接入，默认 `1`，稳定后可试 `2`。
- 日志与错误处理：统一错误结构，日志写入 `week23_project_optimize/logs/app.log`。
- 模型服务化：FastAPI 启动时默认预加载 Qwen3-VL，上传文档后直接复用模型实例推理。
- 压测脚本：默认并发 `1/2/4`，每档 `5` 次。

## 启动

默认启动会初始化并加载 PaddleOCR 和 Qwen3-VL 模型；生产建议保持 `--workers 1`，避免多个 worker 重复占用 GPU。只做接口调试时，可临时设置 `DOC_QA_PRELOAD_OCR=false DOC_QA_PRELOAD_VLM=false`。

```shell
conda run -n doc_parser uvicorn week23_project_optimize.service.app:app   --host 0.0.0.0   --port 9200 \
  --workers 1
```

浏览器打开：

```text
http://127.0.0.1:9200/
```

## API

`POST /api/ask` 支持 multipart form：

- `file`：PDF 文件；或设置 `use_default_pdf=true`
- `question`：问题
- `prompt_type`：`auto/general/table/heading/summary`，默认 `auto`
- `max_images`：最多送入 VLM 的页面图片数，默认 `8`
- `max_new_tokens`：生成长度
- `enable_ppstructure`：可选生成 PPStructure Markdown
- `use_answer_cache`：本次是否启用答案缓存

返回会包含：

- `document_hash`
- `document_cache_hit`
- `answer_cache_hit`
- `prompt_type`
- `selected_pages`
- `timings`

## CLI

```shell
conda run -n doc_parser python -m week23_project_optimize.scripts.client   --url http://127.0.0.1:9200/api/ask   --use-default-pdf   --question "根据这篇文章，总结文章创新点"   --prompt-type summary
```

本地 pipeline，只解析和 OCR，不加载 Qwen3-VL：

```shell
conda run -n doc_parser env DOC_QA_MAX_PAGES=1 python -m week23_project_optimize.scripts.run_pipeline   --pdf docs/notes/BLIP.pdf   --parse-only
```


## 单元测试

这些测试不加载 OCR 或 Qwen3-VL：

```shell
conda run -n doc_parser env PYTHONPATH=. pytest week23_project_optimize/tests
```

## Smoke Test

默认验证文档缓存：第一次 parse 可能 miss，第二次必须 hit。

```shell
conda run -n doc_parser env DOC_QA_MAX_PAGES=1 python -m week23_project_optimize.scripts.smoke_test
```

包含 Qwen3-VL 生成：

```shell
conda run -n doc_parser python -m week23_project_optimize.scripts.smoke_test   --run-vlm   --max-images 2   --max-new-tokens 128
```

## 压测

默认并发 `1/2/4`，每档 `5` 次，默认使用 `docs/notes/BLIP.pdf`：

```shell
conda run -n doc_parser python -m week23_project_optimize.scripts.benchmark   --url http://127.0.0.1:9200/api/ask   --use-default-pdf   --max-images 2   --max-new-tokens 128
```

如果只想测试缓存命中路径，可加：

```shell
--use-answer-cache
```

## 配置

- `DOC_QA_MODEL_PATH`：默认 `models/Qwen3-VL-8B-Instruct`
- `DOC_QA_DEFAULT_PDF`：默认 `docs/notes/BLIP.pdf`
- `DOC_QA_OUTPUT_DIR`：默认 `week23_project_optimize/outputs`
- `DOC_QA_CACHE_DIR`：默认 `week23_project_optimize/cache`
- `DOC_QA_LOG_DIR`：默认 `week23_project_optimize/logs`
- `DOC_QA_DPI`：PDF 渲染 DPI，默认 `180`
- `DOC_QA_OCR_LANG`：OCR 语言，默认 `ch`
- `DOC_QA_OCR_WORKERS`：OCR workers，默认 `1`
- `DOC_QA_MAX_IMAGES`：最多送入 VLM 的页面图片数，默认 `8`
- `DOC_QA_PROMPT_TYPE`：默认 `auto`
- `DOC_QA_ENABLE_DOCUMENT_CACHE`：默认 `true`
- `DOC_QA_ENABLE_ANSWER_CACHE`：默认 `false`
- `DOC_QA_PRELOAD_VLM`：默认 `true`，FastAPI 启动时预加载 Qwen3-VL

## 已知限制

- OCR 并发默认仍为 `1`，因为 PaddleOCR 多 worker 会重复加载模型，需结合机器资源再调优。
- 答案缓存适合 demo 或确定性生成；如果调整 prompt、max token、temperature，cache key 会变化。
- 默认页筛选是轻量关键词匹配，不是向量检索；后续可加 embedding/rerank。
