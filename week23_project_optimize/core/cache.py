"""Document and answer cache helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import stable_json_sha256
from .schemas import DocumentArtifacts, OCRBlock, OCRPage, PageImage


class DocumentCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def document_dir(self, document_hash: str) -> Path:
        return self.cache_dir / document_hash

    def has_document(self, document_hash: str) -> bool:
        return (self.document_dir(document_hash) / "artifacts.json").exists()

    def answer_key(self, payload: dict[str, Any]) -> str:
        return stable_json_sha256(payload)

    def answer_path(self, document_hash: str, answer_key: str) -> Path:
        return self.document_dir(document_hash) / "answers" / f"{answer_key}.json"

    def load_answer(self, document_hash: str, answer_key: str) -> dict[str, Any] | None:
        path = self.answer_path(document_hash, answer_key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_answer(self, document_hash: str, answer_key: str, payload: dict[str, Any]) -> Path:
        path = self.answer_path(document_hash, answer_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.document_dir(document_hash) / "answer.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def artifacts_from_json(path: Path) -> DocumentArtifacts:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = [
        PageImage(
            page_index=item["page_index"],
            page_number=item["page_number"],
            path=Path(item["path"]),
            width=item["width"],
            height=item["height"],
        )
        for item in data["pages"]
    ]
    ocr_pages = []
    for page in data["ocr_pages"]:
        blocks = [
            OCRBlock(
                page_number=block["page_number"],
                text=block["text"],
                box=block.get("box") or [],
                score=block.get("score"),
            )
            for block in page.get("blocks", [])
        ]
        ocr_pages.append(
            OCRPage(
                page_number=page["page_number"],
                image_path=Path(page["image_path"]),
                width=page["width"],
                height=page["height"],
                blocks=blocks,
            )
        )
    markdown_path = data.get("markdown_path")
    return DocumentArtifacts(
        request_id=data["request_id"],
        work_dir=Path(data["work_dir"]),
        input_pdf=Path(data["input_pdf"]),
        pages=pages,
        ocr_pages=ocr_pages,
        ocr_json_path=Path(data["ocr_json_path"]),
        markdown_path=Path(markdown_path) if markdown_path else None,
    )
