"""FastAPI app for PDF upload, OCR, and Qwen3-VL document QA."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from week22_project_core_impl.core.config import PROJECT_ROOT, Settings
from week22_project_core_impl.core.pipeline import DocumentQAPipeline


class RequestQueue:
    def __init__(self, max_concurrency: int, timeout_s: float) -> None:
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._timeout_s = timeout_s

    @asynccontextmanager
    async def slot(self):
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout_s)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=503, detail="request queue timeout") from exc
        try:
            yield
        finally:
            self._semaphore.release()


def _validate_pdf_upload(file: UploadFile, settings: Settings) -> None:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF upload is supported")
    content_type = file.content_type or ""
    if content_type and "pdf" not in content_type and content_type != "application/octet-stream":
        raise HTTPException(status_code=400, detail=f"invalid content type: {content_type}")


async def _save_upload(file: UploadFile, settings: Settings) -> Path:
    _validate_pdf_upload(file, settings)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dst = settings.upload_dir / f"{int(time.time() * 1000)}_{Path(file.filename).name}"
    size = 0
    with dst.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_file_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"file is larger than {settings.max_file_mb} MB")
            f.write(chunk)
    return dst


def _vlm_loaded(pipeline: DocumentQAPipeline) -> bool:
    return pipeline.vlm.model is not None and pipeline.vlm.processor is not None


def _ocr_loaded(pipeline: DocumentQAPipeline) -> bool:
    return pipeline.ocr.loaded


def _preload_models_if_enabled(settings: Settings, pipeline: DocumentQAPipeline) -> None:
    if settings.preload_ocr:
        pipeline.ocr.load()
    if settings.preload_vlm:
        pipeline.vlm.load()


def create_app(settings: Settings | None = None, pipeline: DocumentQAPipeline | None = None) -> FastAPI:
    settings = settings or Settings()
    pipeline = pipeline or DocumentQAPipeline(settings)

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        await asyncio.to_thread(_preload_models_if_enabled, settings, pipeline)
        yield

    app = FastAPI(title="Week22 Document QA Alpha", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.queue = RequestQueue(settings.max_concurrency, settings.queue_timeout_s)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = PROJECT_ROOT / "frontend" / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": str(settings.model_path),
            "output_dir": str(settings.output_dir),
            "default_pdf": str(settings.default_pdf),
            "max_images": settings.max_images,
            "ocr_device": settings.ocr_device,
            "preload_ocr": settings.preload_ocr,
            "preload_vlm": settings.preload_vlm,
            "ocr_loaded": _ocr_loaded(app.state.pipeline),
            "vlm_loaded": _vlm_loaded(app.state.pipeline),
        }

    @app.get("/outputs/{request_id}/{filename}")
    async def output_file(request_id: str, filename: str) -> FileResponse:
        allowed = {"ocr.json", "answer.json", "artifacts.json", "input.pdf"}
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="file not found")
        path = settings.output_dir / request_id / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(path)

    @app.post("/api/parse")
    async def parse_document(
        file: UploadFile = File(...),
        enable_ppstructure: bool = Form(default=False),
    ) -> dict[str, Any]:
        saved_pdf = await _save_upload(file, settings)
        async with app.state.queue.slot():
            artifacts = await asyncio.to_thread(
                app.state.pipeline.parse_pdf,
                saved_pdf,
                None,
                enable_ppstructure,
            )
        return artifacts.to_json()

    @app.post("/api/ask")
    async def ask_document(
        question: str = Form(...),
        file: UploadFile | None = File(default=None),
        use_default_pdf: bool = Form(default=False),
        max_images: int | None = Form(default=None),
        max_new_tokens: int | None = Form(default=None),
        enable_ppstructure: bool = Form(default=False),
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="question is required")

        if use_default_pdf:
            if not settings.default_pdf.exists():
                raise HTTPException(status_code=404, detail=f"default PDF not found: {settings.default_pdf}")
            pdf_path = settings.default_pdf
        elif file is not None:
            pdf_path = await _save_upload(file, settings)
        else:
            raise HTTPException(status_code=400, detail="upload a PDF or set use_default_pdf=true")

        async with app.state.queue.slot():
            result = await asyncio.to_thread(
                app.state.pipeline.ask,
                pdf_path,
                question,
                None,
                max_images,
                max_new_tokens,
                enable_ppstructure,
            )
        return result

    return app


app = create_app()

