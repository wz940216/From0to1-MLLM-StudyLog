"""End-to-end document parsing and question answering pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .document_store import create_request_dir, persist_input_pdf
from .ocr_engine import PaddleOCREngine, run_basic_ocr, run_ppstructure_markdown
from .pdf_splitter import split_pdf_to_images
from .schemas import DocumentArtifacts
from .vlm_engine import QwenVLDocumentQA


class DocumentQAPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ocr = PaddleOCREngine(settings.ocr_lang, settings.ocr_device)
        self.vlm = QwenVLDocumentQA(settings.model_path, settings.device_map)

    def parse_pdf(self, pdf_path: Path, request_id: str | None = None, enable_ppstructure: bool | None = None) -> DocumentArtifacts:
        request_id, work_dir = create_request_dir(self.settings.output_dir, request_id)
        input_pdf = persist_input_pdf(pdf_path, work_dir)
        pages = split_pdf_to_images(
            input_pdf,
            work_dir / "pages",
            dpi=self.settings.dpi,
            max_pages=self.settings.max_pages,
        )
        ocr_json_path = work_dir / "ocr.json"
        ocr_pages = run_basic_ocr(
            pages,
            ocr_json_path,
            lang=self.settings.ocr_lang,
            device=self.settings.ocr_device,
            ocr_engine=self.ocr,
        )

        markdown_path = None
        should_run_ppstructure = self.settings.enable_ppstructure if enable_ppstructure is None else enable_ppstructure
        if should_run_ppstructure:
            markdown_path = run_ppstructure_markdown(input_pdf, work_dir / "ppstructure", lang=self.settings.ocr_lang)

        artifacts = DocumentArtifacts(
            request_id=request_id,
            work_dir=work_dir,
            input_pdf=input_pdf,
            pages=pages,
            ocr_pages=ocr_pages,
            ocr_json_path=ocr_json_path,
            markdown_path=markdown_path,
        )
        (work_dir / "artifacts.json").write_text(
            json.dumps(artifacts.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return artifacts

    def ask(
        self,
        pdf_path: Path,
        question: str,
        request_id: str | None = None,
        max_images: int | None = None,
        max_new_tokens: int | None = None,
        enable_ppstructure: bool | None = None,
    ) -> dict:
        artifacts = self.parse_pdf(pdf_path, request_id=request_id, enable_ppstructure=enable_ppstructure)
        result = self.vlm.answer(
            artifacts=artifacts,
            question=question,
            max_images=self.settings.max_images if max_images is None else max_images,
            max_input_chars=self.settings.max_input_chars,
            max_new_tokens=self.settings.max_new_tokens if max_new_tokens is None else max_new_tokens,
            temperature=self.settings.temperature,
        )
        answer_json = {
            "request_id": artifacts.request_id,
            "question": question,
            "answer": result["answer"],
            "selected_pages": result["selected_pages"],
            "artifacts": artifacts.to_json(),
        }
        (artifacts.work_dir / "answer.json").write_text(
            json.dumps(answer_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return answer_json

