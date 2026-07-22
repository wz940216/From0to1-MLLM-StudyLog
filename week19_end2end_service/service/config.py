"""Runtime configuration for the week19 FastAPI service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


@dataclass(frozen=True)
class Settings:
    """Small env-driven settings object.

    Environment variables are prefixed with ``MINILLAVA_`` so the service can be
    configured from shell scripts without introducing a separate config file.
    """

    vllm_base_url: str = os.getenv("MINILLAVA_VLLM_BASE_URL", "http://127.0.0.1:8000")
    model: str = os.getenv("MINILLAVA_MODEL", "minillava")
    request_timeout_s: float = _get_float("MINILLAVA_REQUEST_TIMEOUT_S", 120.0)
    max_image_mb: int = _get_int("MINILLAVA_MAX_IMAGE_MB", 8)
    max_image_side: int = _get_int("MINILLAVA_MAX_IMAGE_SIDE", 1024)
    max_concurrency: int = _get_int("MINILLAVA_MAX_CONCURRENCY", 2)
    queue_timeout_s: float = _get_float("MINILLAVA_QUEUE_TIMEOUT_S", 30.0)
    rate_limit_per_min: int = _get_int("MINILLAVA_RATE_LIMIT_PER_MIN", 30)
    default_max_tokens: int = _get_int("MINILLAVA_DEFAULT_MAX_TOKENS", 128)
    default_temperature: float = _get_float("MINILLAVA_DEFAULT_TEMPERATURE", 0.0)

