"""Structured API error helpers."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException


@dataclass(frozen=True)
class ErrorCode:
    code: str
    status_code: int


BAD_REQUEST = ErrorCode("BAD_REQUEST", 400)
FILE_TOO_LARGE = ErrorCode("FILE_TOO_LARGE", 413)
QUEUE_TIMEOUT = ErrorCode("QUEUE_TIMEOUT", 503)
PARSE_FAILED = ErrorCode("PARSE_FAILED", 500)
VLM_FAILED = ErrorCode("VLM_FAILED", 500)
NOT_FOUND = ErrorCode("NOT_FOUND", 404)


def raise_api_error(error: ErrorCode, message: str, request_id: str | None = None, stage: str | None = None) -> None:
    raise HTTPException(
        status_code=error.status_code,
        detail={
            "error": {
                "code": error.code,
                "message": message,
                "request_id": request_id,
                "stage": stage,
            }
        },
    )
