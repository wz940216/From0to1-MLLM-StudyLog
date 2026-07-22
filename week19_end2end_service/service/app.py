"""FastAPI application for an end-to-end MiniLLaVA service."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .config import Settings
from .image_io import prepare_upload_image
from .limiter import FixedWindowRateLimiter, RequestQueue
from .vllm_client import VllmChatClient


logger = logging.getLogger("week19_end2end_service")


def _build_messages(
    question: str,
    image_data_url: str,
    image_uuid: str,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    # 这里构造的是 vLLM OpenAI-compatible Chat Completions 接口需要的 messages。
    # 对多模态输入，content 不是普通字符串，而是 text/image_url 组成的列表。
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                # vLLM 0.10.x 会把 uuid 当作图片缓存和 mm_hashes 的稳定标识。
                # 自定义 Transformers 多模态模型如果不传 uuid，可能得到 None hash 并触发 400。
                {"type": "image_url", "uuid": image_uuid, "image_url": {"url": image_data_url}},
                {"type": "text", "text": question},
            ],
        }
    )
    return messages


def create_app(settings: Settings | None = None, backend: Any | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        yield
        # 服务关闭时释放 httpx 连接池，避免进程退出前残留未关闭连接。
        close = getattr(app_.state.backend, "close", None)
        if close is not None:
            await close()

    app = FastAPI(title="MiniLLaVA End-to-End Service", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    # rate_limiter、queue、backend 都挂在 app.state 上，保证每个 worker 进程内复用同一份状态。
    app.state.rate_limiter = FixedWindowRateLimiter(settings.rate_limit_per_min)
    app.state.queue = RequestQueue(settings.max_concurrency, settings.queue_timeout_s)
    app.state.backend = backend or VllmChatClient(
        base_url=settings.vllm_base_url,
        model=settings.model,
        timeout_s=settings.request_timeout_s,
    )


    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": settings.model,
            "vllm_base_url": settings.vllm_base_url,
            "max_concurrency": settings.max_concurrency,
            "rate_limit_per_min": settings.rate_limit_per_min,
        }

    @app.post("/chat")
    async def chat(
        request: Request,
        image: UploadFile = File(...),
        question: str = Form(...),
        system_prompt: str | None = Form(default=None),
        max_tokens: int | None = Form(default=None),
        temperature: float | None = Form(default=None),
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        # 先按客户端 IP 做轻量限流；这里只是单进程内存限流，生产环境可替换成 Redis。
        client_host = request.client.host if request.client else "unknown"
        await app.state.rate_limiter.check(client_host)

        request_id = uuid.uuid4().hex
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # 用队列限制同时进入 vLLM 的请求数，避免 GPU 推理服务被瞬时并发打满。
        async with app.state.queue.slot():
            phase_start = time.perf_counter()
            # 读取 multipart 上传的图片，完成校验、缩放、编码，并生成稳定 uuid。
            prepared_image = await prepare_upload_image(
                image,
                max_image_mb=settings.max_image_mb,
                max_image_side=settings.max_image_side,
            )
            timings["preprocess_ms"] = round((time.perf_counter() - phase_start) * 1000, 3)

            phase_start = time.perf_counter()
            # 把业务接口的 question/image 转成 vLLM 能理解的 OpenAI 多模态 messages。
            messages = _build_messages(question, prepared_image.data_url, prepared_image.uuid, system_prompt)
            timings["prompt_ms"] = round((time.perf_counter() - phase_start) * 1000, 3)

            phase_start = time.perf_counter()
            # 真正的模型推理由 vLLM 服务完成；FastAPI 只做 HTTP 转发和结果解析。
            answer = await app.state.backend.generate(
                messages=messages,
                max_tokens=max_tokens or settings.default_max_tokens,
                temperature=settings.default_temperature if temperature is None else temperature,
            )
            timings["generate_ms"] = round((time.perf_counter() - phase_start) * 1000, 3)

        timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 3)
        # 记录分阶段耗时，方便区分慢在图片预处理、排队等待还是 vLLM 生成。
        logger.info(
            "request_id=%s client=%s image=%sx%s bytes=%s timings=%s",
            request_id,
            client_host,
            prepared_image.width,
            prepared_image.height,
            prepared_image.bytes_size,
            timings,
        )
        return {
            "request_id": request_id,
            "answer": answer,
            "model": settings.model,
            "image": {
                "width": prepared_image.width,
                "height": prepared_image.height,
                "mime_type": prepared_image.mime_type,
            },
            "timings": timings,
        }

    return app


app = create_app()

