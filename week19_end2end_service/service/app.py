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
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
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
        close = getattr(app_.state.backend, "close", None)
        if close is not None:
            await close()

    app = FastAPI(title="MiniLLaVA End-to-End Service", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
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

        client_host = request.client.host if request.client else "unknown"
        await app.state.rate_limiter.check(client_host)

        request_id = uuid.uuid4().hex
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        async with app.state.queue.slot():
            phase_start = time.perf_counter()
            prepared_image = await prepare_upload_image(
                image,
                max_image_mb=settings.max_image_mb,
                max_image_side=settings.max_image_side,
            )
            timings["preprocess_ms"] = round((time.perf_counter() - phase_start) * 1000, 3)

            phase_start = time.perf_counter()
            messages = _build_messages(question, prepared_image.data_url, prepared_image.uuid, system_prompt)
            timings["prompt_ms"] = round((time.perf_counter() - phase_start) * 1000, 3)

            phase_start = time.perf_counter()
            answer = await app.state.backend.generate(
                messages=messages,
                max_tokens=max_tokens or settings.default_max_tokens,
                temperature=settings.default_temperature if temperature is None else temperature,
            )
            timings["generate_ms"] = round((time.perf_counter() - phase_start) * 1000, 3)

        timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 3)
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

