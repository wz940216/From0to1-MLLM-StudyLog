"""Shared dataclasses for document parsing results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PageImage:
    page_index: int
    page_number: int
    path: Path
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass
class OCRBlock:
    page_number: int
    text: str
    box: list[list[float]]
    score: float | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRPage:
    page_number: int
    image_path: Path
    width: int
    height: int
    blocks: list[OCRBlock]

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text).strip()

    def to_json(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "blocks": [block.to_json() for block in self.blocks],
        }


@dataclass
class DocumentArtifacts:
    request_id: str
    work_dir: Path
    input_pdf: Path
    pages: list[PageImage]
    ocr_pages: list[OCRPage]
    ocr_json_path: Path
    markdown_path: Path | None = None

    @property
    def full_text(self) -> str:
        parts = []
        for page in self.ocr_pages:
            if page.text:
                parts.append(f"[Page {page.page_number}]\n{page.text}")
        return "\n\n".join(parts).strip()

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "work_dir": str(self.work_dir),
            "input_pdf": str(self.input_pdf),
            "pages": [page.to_json() for page in self.pages],
            "ocr_json_path": str(self.ocr_json_path),
            "markdown_path": str(self.markdown_path) if self.markdown_path else None,
            "ocr_pages": [page.to_json() for page in self.ocr_pages],
        }

