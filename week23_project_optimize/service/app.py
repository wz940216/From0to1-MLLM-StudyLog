"""FastAPI app for optimized PDF OCR and Qwen3-VL document QA."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from week23_project_optimize.core.config import PROJECT_ROOT, Settings
from week23_project_optimize.core.pipeline import DocumentQAPipeline
from week23_project_optimize.service.errors import BAD_REQUEST, FILE_TOO_LARGE, NOT_FOUND, QUEUE_TIMEOUT, raise_api_error
from week23_project_optimize.service.logging_config import setup_logging


class RequestQueue:
    def __init__(self, max_concurrency: int, timeout_s: float) -> None:
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._timeout_s = timeout_s

    @asynccontextmanager
    async def slot(self):
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            raise_api_error(QUEUE_TIMEOUT, "request queue timeout", stage="queue")
        try:
            yield
        finally:
            self._semaphore.release()


def _validate_pdf_upload(file: UploadFile, settings: Settings) -> None:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise_api_error(BAD_REQUEST, "only PDF upload is supported", stage="upload")
    content_type = file.content_type or ""
    if content_type and "pdf" not in content_type and content_type != "application/octet-stream":
        raise_api_error(BAD_REQUEST, f"invalid content type: {content_type}", stage="upload")


async def _save_upload(file: UploadFile, settings: Settings) -> Path:
    _validate_pdf_upload(file, settings)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dst = settings.upload_dir / f"{int(time.time() * 1000)}_{Path(file.filename).name}"
    size = 0
    with dst.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_file_mb * 1024 * 1024:
                raise_api_error(FILE_TOO_LARGE, f"file is larger than {settings.max_file_mb} MB", stage="upload")
            f.write(chunk)
    return dst


def _find_artifact(settings: Settings, request_id: str, filename: str) -> Path | None:
    candidates = [
        settings.output_dir / request_id / filename,
        settings.cache_dir / request_id / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _vlm_loaded(pipeline: DocumentQAPipeline) -> bool:
    return pipeline.vlm.model is not None and pipeline.vlm.processor is not None


def _ocr_loaded(pipeline: DocumentQAPipeline) -> bool:
    return pipeline.ocr.loaded


def _preload_models_if_enabled(settings: Settings, pipeline: DocumentQAPipeline, logger) -> None:
    if settings.preload_ocr:
        logger.info("preloading PaddleOCR lang=%s device=%s", settings.ocr_lang, settings.ocr_device)
        pipeline.ocr.load()
        logger.info("PaddleOCR preload finished")
    if settings.preload_vlm:
        logger.info("preloading Qwen-VL model path=%s device_map=%s", settings.model_path, settings.device_map)
        pipeline.vlm.load()
        logger.info("Qwen-VL model preload finished")


def create_app(settings: Settings | None = None, pipeline: DocumentQAPipeline | None = None) -> FastAPI:
    settings = settings or Settings()
    logger = setup_logging(settings.log_dir)
    pipeline = pipeline or DocumentQAPipeline(settings)

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        await asyncio.to_thread(_preload_models_if_enabled, settings, pipeline, logger)
        yield

    app = FastAPI(title="Week23 Document QA Beta", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.queue = RequestQueue(settings.max_concurrency, settings.queue_timeout_s)
    app.state.logger = logger

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "HTTP_ERROR", "message": str(exc.detail), "request_id": None, "stage": None}},
        )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = PROJECT_ROOT / "frontend" / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "week23-beta",
            "model_path": str(settings.model_path),
            "cache_dir": str(settings.cache_dir),
            "output_dir": str(settings.output_dir),
            "default_pdf": str(settings.default_pdf),
            "max_images": settings.max_images,
            "ocr_workers": settings.ocr_workers,
            "ocr_device": settings.ocr_device,
            "document_cache": settings.enable_document_cache,
            "answer_cache": settings.enable_answer_cache,
            "preload_ocr": settings.preload_ocr,
            "preload_vlm": settings.preload_vlm,
            "ocr_loaded": _ocr_loaded(app.state.pipeline),
            "vlm_loaded": _vlm_loaded(app.state.pipeline),
        }

    @app.get("/outputs/{request_id}/{filename}")
    async def output_file(request_id: str, filename: str) -> FileResponse:
        allowed = {"ocr.json", "answer.json", "artifacts.json", "input.pdf"}
        if filename not in allowed:
            raise_api_error(NOT_FOUND, "file not found", request_id=request_id, stage="download")
        path = _find_artifact(settings, request_id, filename)
        if path is None:
            raise_api_error(NOT_FOUND, "file not found", request_id=request_id, stage="download")
        return FileResponse(path)

    @app.post("/api/parse")
    async def parse_document(
        file: UploadFile = File(...),
        enable_ppstructure: bool = Form(default=False),
    ) -> dict[str, Any]:
        saved_pdf = await _save_upload(file, settings)
        async with app.state.queue.slot():
            artifacts = await asyncio.to_thread(app.state.pipeline.parse_pdf, saved_pdf, None, enable_ppstructure)
        meta = app.state.pipeline.last_parse_meta
        logger.info(
            "parse request_id=%s document_hash=%s cache_hit=%s pages=%s timings=%s",
            artifacts.request_id,
            meta.get("document_hash"),
            meta.get("cache_hit"),
            len(artifacts.pages),
            meta.get("timings"),
        )
        payload = artifacts.to_json()
        payload["document_hash"] = meta.get("document_hash")
        payload["document_cache_hit"] = meta.get("cache_hit")
        payload["timings"] = meta.get("timings")
        return payload

    @app.post("/api/ask")
    async def ask_document(
        question: str = Form(...),
        file: UploadFile | None = File(default=None),
        use_default_pdf: bool = Form(default=False),
        max_images: int | None = Form(default=None),
        max_new_tokens: int | None = Form(default=None),
        enable_ppstructure: bool = Form(default=False),
        prompt_type: str | None = Form(default=None),
        use_answer_cache: bool | None = Form(default=None),
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise_api_error(BAD_REQUEST, "question is required", stage="validate")

        if use_default_pdf:
            if not settings.default_pdf.exists():
                raise_api_error(NOT_FOUND, f"default PDF not found: {settings.default_pdf}", stage="load_pdf")
            pdf_path = settings.default_pdf
        elif file is not None:
            pdf_path = await _save_upload(file, settings)
        else:
            raise_api_error(BAD_REQUEST, "upload a PDF or set use_default_pdf=true", stage="validate")

        async with app.state.queue.slot():
            result = await asyncio.to_thread(
                app.state.pipeline.ask,
                pdf_path,
                question,
                None,
                max_images,
                max_new_tokens,
                enable_ppstructure,
                prompt_type,
                use_answer_cache,
            )
        logger.info(
            "ask request_id=%s document_hash=%s doc_cache=%s answer_cache=%s prompt=%s pages=%s timings=%s",
            result.get("request_id"),
            result.get("document_hash"),
            result.get("document_cache_hit"),
            result.get("answer_cache_hit"),
            result.get("prompt_type"),
            result.get("selected_pages"),
            result.get("timings"),
        )
        return result

    return app


app = create_app()
