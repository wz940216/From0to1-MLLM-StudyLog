"""Optimized document parsing and question answering pipeline for week23 beta."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .cache import DocumentCache, artifacts_from_json
from .config import Settings
from .hashing import file_sha256
from .ocr_engine import PaddleOCREngine, run_basic_ocr, run_ppstructure_markdown
from .pdf_splitter import split_pdf_to_images
from .schemas import DocumentArtifacts
from .timings import TimingRecorder
from .vlm_engine import QwenVLDocumentQA


class DocumentQAPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = DocumentCache(settings.cache_dir)
        self.ocr = PaddleOCREngine(settings.ocr_lang, settings.ocr_device)
        self.vlm = QwenVLDocumentQA(settings.model_path, settings.device_map)
        self.last_parse_meta: dict = {}

    def _write_artifacts(self, artifacts: DocumentArtifacts) -> None:
        (artifacts.work_dir / "artifacts.json").write_text(
            json.dumps(artifacts.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def parse_pdf(
        self,
        pdf_path: Path,
        request_id: str | None = None,
        enable_ppstructure: bool | None = None,
    ) -> DocumentArtifacts:
        timer = TimingRecorder()
        pdf_path = pdf_path.resolve()
        with timer.track("hash_ms"):
            document_hash = file_sha256(pdf_path)

        cache_enabled = self.settings.enable_document_cache
        work_dir = self.cache.document_dir(document_hash) if cache_enabled else self.settings.output_dir / (request_id or document_hash)
        artifacts_path = work_dir / "artifacts.json"
        should_run_ppstructure = self.settings.enable_ppstructure if enable_ppstructure is None else enable_ppstructure

        cache_hit = cache_enabled and artifacts_path.exists()
        if cache_hit:
            with timer.track("load_cache_ms"):
                artifacts = artifacts_from_json(artifacts_path)
            if should_run_ppstructure and artifacts.markdown_path is None:
                with timer.track("ppstructure_ms"):
                    markdown_path = run_ppstructure_markdown(artifacts.input_pdf, work_dir / "ppstructure", lang=self.settings.ocr_lang)
                artifacts.markdown_path = markdown_path
                self._write_artifacts(artifacts)
            self.last_parse_meta = {
                "document_hash": document_hash,
                "cache_hit": True,
                "timings": timer.timings,
            }
            return artifacts

        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "pages").mkdir(exist_ok=True)
        (work_dir / "ppstructure").mkdir(exist_ok=True)
        input_pdf = work_dir / "input.pdf"
        with timer.track("copy_input_ms"):
            shutil.copy2(pdf_path, input_pdf)
        with timer.track("split_pdf_ms"):
            pages = split_pdf_to_images(
                input_pdf,
                work_dir / "pages",
                dpi=self.settings.dpi,
                max_pages=self.settings.max_pages,
            )
        ocr_json_path = work_dir / "ocr.json"
        with timer.track("ocr_ms"):
            ocr_pages = run_basic_ocr(
                pages,
                ocr_json_path,
                lang=self.settings.ocr_lang,
                device=self.settings.ocr_device,
                ocr_engine=self.ocr,
                workers=self.settings.ocr_workers,
            )

        markdown_path = None
        if should_run_ppstructure:
            with timer.track("ppstructure_ms"):
                markdown_path = run_ppstructure_markdown(input_pdf, work_dir / "ppstructure", lang=self.settings.ocr_lang)

        artifacts = DocumentArtifacts(
            request_id=document_hash,
            work_dir=work_dir,
            input_pdf=input_pdf,
            pages=pages,
            ocr_pages=ocr_pages,
            ocr_json_path=ocr_json_path,
            markdown_path=markdown_path,
        )
        self._write_artifacts(artifacts)
        self.last_parse_meta = {
            "document_hash": document_hash,
            "cache_hit": False,
            "timings": timer.timings,
        }
        return artifacts

    def ask(
        self,
        pdf_path: Path,
        question: str,
        request_id: str | None = None,
        max_images: int | None = None,
        max_new_tokens: int | None = None,
        enable_ppstructure: bool | None = None,
        prompt_type: str | None = None,
        use_answer_cache: bool | None = None,
    ) -> dict:
        timer = TimingRecorder()
        with timer.track("parse_total_ms"):
            artifacts = self.parse_pdf(pdf_path, request_id=request_id, enable_ppstructure=enable_ppstructure)
        parse_meta = dict(self.last_parse_meta)
        document_hash = parse_meta["document_hash"]
        effective_max_images = self.settings.max_images if max_images is None else max_images
        effective_max_new_tokens = self.settings.max_new_tokens if max_new_tokens is None else max_new_tokens
        effective_prompt_type = self.settings.prompt_type if prompt_type is None else prompt_type
        answer_cache_enabled = self.settings.enable_answer_cache if use_answer_cache is None else use_answer_cache
        answer_cache_key = self.cache.answer_key(
            {
                "document_hash": document_hash,
                "question": question,
                "max_images": effective_max_images,
                "max_input_chars": self.settings.max_input_chars,
                "max_new_tokens": effective_max_new_tokens,
                "temperature": self.settings.temperature,
                "prompt_type": effective_prompt_type,
            }
        )

        if answer_cache_enabled:
            with timer.track("answer_cache_lookup_ms"):
                cached = self.cache.load_answer(document_hash, answer_cache_key)
            if cached is not None:
                cached["answer_cache_hit"] = True
                cached["document_cache_hit"] = parse_meta["cache_hit"]
                cached["timings"] = {**parse_meta.get("timings", {}), **timer.timings}
                return cached

        with timer.track("vlm_generate_ms"):
            result = self.vlm.answer(
                artifacts=artifacts,
                question=question,
                max_images=effective_max_images,
                max_input_chars=self.settings.max_input_chars,
                max_new_tokens=effective_max_new_tokens,
                temperature=self.settings.temperature,
                prompt_type=effective_prompt_type,
            )

        answer_json = {
            "request_id": document_hash,
            "document_hash": document_hash,
            "question": question,
            "answer": result["answer"],
            "prompt_type": result.get("prompt_type", effective_prompt_type),
            "selected_pages": result["selected_pages"],
            "document_cache_hit": parse_meta["cache_hit"],
            "answer_cache_hit": False,
            "answer_cache_key": answer_cache_key,
            "artifacts": artifacts.to_json(),
            "timings": {**parse_meta.get("timings", {}), **timer.timings},
        }
        if answer_cache_enabled:
            self.cache.save_answer(document_hash, answer_cache_key, answer_json)
        else:
            (artifacts.work_dir / "answer.json").write_text(
                json.dumps(answer_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return answer_json
