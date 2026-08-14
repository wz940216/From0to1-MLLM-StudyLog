"""Basic OCR extraction and optional PPStructure markdown generation."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .schemas import OCRBlock, OCRPage, PageImage


def _prepare_paddle_cache() -> None:
    cache_home = os.environ.get("PADDLE_PDX_CACHE_HOME") or os.environ.get("PADDLEX_HOME") or str(Path("models/.paddlex"))
    os.environ["PADDLE_PDX_CACHE_HOME"] = cache_home
    os.environ["PADDLEX_HOME"] = cache_home
    Path(cache_home, "official_models").mkdir(parents=True, exist_ok=True)


def _make_paddle_ocr(lang: str, device: str = "gpu"):
    _prepare_paddle_cache()
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(
            lang=lang,
            ocr_version="PP-OCRv5",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
    except TypeError:
        return PaddleOCR(lang=lang, use_angle_cls=False)


def _normalize_box(raw_box: Any) -> list[list[float]]:
    if raw_box is None:
        return []
    return [[float(point[0]), float(point[1])] for point in raw_box]


def _parse_old_result(page_number: int, result: Any) -> list[OCRBlock]:
    blocks: list[OCRBlock] = []
    if not result:
        return blocks
    lines = result[0] if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list) else result
    for line in lines or []:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        box = _normalize_box(line[0])
        rec = line[1]
        text = ""
        score = None
        if isinstance(rec, (list, tuple)) and rec:
            text = str(rec[0])
            if len(rec) > 1:
                score = float(rec[1])
        elif isinstance(rec, str):
            text = rec
        if text:
            blocks.append(OCRBlock(page_number=page_number, text=text, box=box, score=score))
    return blocks


def _parse_dict_result(page_number: int, result: dict[str, Any]) -> list[OCRBlock]:
    texts = result.get("rec_texts") or result.get("texts") or []
    scores = result.get("rec_scores") or result.get("scores") or []
    boxes = result.get("rec_polys") or result.get("dt_polys") or result.get("boxes") or []
    blocks: list[OCRBlock] = []
    for index, text in enumerate(texts):
        if not text:
            continue
        score = float(scores[index]) if index < len(scores) and scores[index] is not None else None
        box = _normalize_box(boxes[index]) if index < len(boxes) else []
        blocks.append(OCRBlock(page_number=page_number, text=str(text), box=box, score=score))
    return blocks


def _run_single_page(ocr: Any, page: PageImage) -> list[OCRBlock]:
    if hasattr(ocr, "predict"):
        result = ocr.predict(str(page.path))
    else:
        result = ocr.ocr(str(page.path), cls=True)
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return _parse_dict_result(page.page_number, result[0])
    if isinstance(result, dict):
        return _parse_dict_result(page.page_number, result)
    return _parse_old_result(page.page_number, result)


def _ocr_page_to_result(ocr, page: PageImage) -> OCRPage:
    blocks = _run_single_page(ocr, page)
    return OCRPage(
        page_number=page.page_number,
        image_path=page.path,
        width=page.width,
        height=page.height,
        blocks=blocks,
    )


def _write_ocr_json(ocr_pages: list[OCRPage], output_json_path: Path) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps({"pages": [page.to_json() for page in ocr_pages]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class PaddleOCREngine:
    """Reusable PaddleOCR wrapper.

    A single PaddleOCR instance is loaded at FastAPI startup and protected by a
    lock during inference. PaddleOCR/GPU execution is expensive to initialize and
    is not assumed to be safe for concurrent access from multiple requests.
    """

    def __init__(self, lang: str = "ch", device: str = "gpu") -> None:
        self.lang = lang
        self.device = device
        self.ocr = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.ocr is not None

    def load(self) -> None:
        if self.ocr is None:
            self.ocr = _make_paddle_ocr(self.lang, self.device)

    def run_pages(self, pages: list[PageImage], output_json_path: Path) -> list[OCRPage]:
        self.load()
        with self._lock:
            ocr_pages = [_ocr_page_to_result(self.ocr, page) for page in pages]
        ocr_pages = sorted(ocr_pages, key=lambda page: page.page_number)
        _write_ocr_json(ocr_pages, output_json_path)
        return ocr_pages


def run_basic_ocr(
    pages: list[PageImage],
    output_json_path: Path,
    lang: str = "ch",
    device: str = "gpu",
    ocr_engine: PaddleOCREngine | None = None,
    workers: int = 1,
) -> list[OCRPage]:
    """Run OCR for rendered pages and persist structured JSON."""
    if ocr_engine is not None:
        return ocr_engine.run_pages(pages, output_json_path)

    ocr = _make_paddle_ocr(lang, device)
    ocr_pages = [_ocr_page_to_result(ocr, page) for page in pages]
    ocr_pages = sorted(ocr_pages, key=lambda page: page.page_number)
    _write_ocr_json(ocr_pages, output_json_path)
    return ocr_pages


def run_ppstructure_markdown(pdf_path: Path, output_dir: Path, lang: str = "ch") -> Path:
    """Optionally generate PPStructure markdown for the source PDF."""
    _prepare_paddle_cache()
    from paddleocr import PPStructureV3

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PPStructureV3(
        engine="transformers",
        lang=lang,
        use_table_recognition=False,
        use_formula_recognition=False,
    )
    output = pipeline.predict(input=str(pdf_path))
    markdown_list = []
    markdown_images = []
    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
    if not isinstance(markdown_text, str):
        markdown_text = markdown_text["markdown_texts"]
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown_text, encoding="utf-8")

    for item in markdown_images:
        for raw_path, image in (item or {}).items():
            image_path = output_dir / raw_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)
    return markdown_path
