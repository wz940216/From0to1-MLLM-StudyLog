"""Persistent artifact storage helpers."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def create_request_dir(output_root: Path, request_id: str | None = None) -> tuple[str, Path]:
    request_id = request_id or uuid.uuid4().hex
    work_dir = output_root / request_id
    work_dir.mkdir(parents=True, exist_ok=False)
    (work_dir / "pages").mkdir()
    (work_dir / "ppstructure").mkdir()
    return request_id, work_dir


def persist_input_pdf(src_pdf: Path, work_dir: Path) -> Path:
    dst = work_dir / "input.pdf"
    shutil.copy2(src_pdf, dst)
    return dst

