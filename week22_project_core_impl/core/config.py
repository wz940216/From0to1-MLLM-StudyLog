"""Runtime configuration for the week22 document QA pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("DOC_QA_MODEL_PATH", ROOT / "models" / "Qwen3-VL-8B-Instruct"))
    output_dir: Path = Path(os.getenv("DOC_QA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    upload_dir: Path = Path(os.getenv("DOC_QA_UPLOAD_DIR", PROJECT_ROOT / "uploads"))
    default_pdf: Path = Path(os.getenv("DOC_QA_DEFAULT_PDF", ROOT / "docs" / "notes" / "BLIP.pdf"))

    dpi: int = _get_int("DOC_QA_DPI", 180)
    ocr_lang: str = os.getenv("DOC_QA_OCR_LANG", "ch")
    ocr_device: str = os.getenv("DOC_QA_OCR_DEVICE", "gpu")
    max_pages: int = _get_int("DOC_QA_MAX_PAGES", 0)
    max_images: int = _get_int("DOC_QA_MAX_IMAGES", 8)
    max_input_chars: int = _get_int("DOC_QA_MAX_INPUT_CHARS", 24000)
    max_file_mb: int = _get_int("DOC_QA_MAX_FILE_MB", 80)
    enable_ppstructure: bool = _get_bool("DOC_QA_ENABLE_PPSTRUCTURE", False)
    preload_vlm: bool = _get_bool("DOC_QA_PRELOAD_VLM", True)
    preload_ocr: bool = _get_bool("DOC_QA_PRELOAD_OCR", True)

    device_map: str = os.getenv("DOC_QA_DEVICE_MAP", "auto")
    max_new_tokens: int = _get_int("DOC_QA_MAX_NEW_TOKENS", 2048)
    temperature: float = _get_float("DOC_QA_TEMPERATURE", 0.0)
    queue_timeout_s: float = _get_float("DOC_QA_QUEUE_TIMEOUT_S", 30.0)
    max_concurrency: int = _get_int("DOC_QA_MAX_CONCURRENCY", 1)

