"""PDF to page image conversion."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .schemas import PageImage


def split_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 180, max_pages: int = 0) -> list[PageImage]:
    """Render a PDF into PNG page images using PyMuPDF."""
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError("PDF splitting requires PyMuPDF. Install it with: pip install pymupdf") from exc

    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF input is supported, got: {pdf_path.suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[PageImage] = []

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        limit = total_pages if max_pages <= 0 else min(max_pages, total_pages)
        for index in range(limit):
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{index + 1:03d}.png"
            pix.save(str(image_path))
            with Image.open(image_path) as image:
                width, height = image.size
            pages.append(
                PageImage(
                    page_index=index,
                    page_number=index + 1,
                    path=image_path,
                    width=width,
                    height=height,
                )
            )
    return pages

