"""Locust load test for the week19 MiniLLaVA FastAPI service.

Target endpoint:
    POST /chat

The request is intentionally the same multipart shape as the real client:
    image=<binary file>, question=<text>, max_tokens=<int>, temperature=<float>
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

from locust import HttpUser, between, events, task


DEFAULT_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "week03_mllm_overview_llava_demo"
    / "code"
    / "image.png"
)

IMAGE_PATH = Path(os.getenv("LOCUST_IMAGE_PATH", str(DEFAULT_IMAGE))).expanduser()
QUESTION = os.getenv("LOCUST_QUESTION", "请描述这张图片。")
MAX_TOKENS = os.getenv("LOCUST_MAX_TOKENS", "64")
TEMPERATURE = os.getenv("LOCUST_TEMPERATURE", "0.0")
REQUEST_TIMEOUT_S = float(os.getenv("LOCUST_REQUEST_TIMEOUT_S", "180"))
HEALTH_WEIGHT = int(os.getenv("LOCUST_HEALTH_WEIGHT", "1"))
CHAT_WEIGHT = int(os.getenv("LOCUST_CHAT_WEIGHT", "20"))

QUESTIONS = [
    q.strip()
    for q in os.getenv(
        "LOCUST_QUESTIONS",
        (
            "请描述这张图片。|"
            "这张图里有什么主要物体？|"
            "请用一句话概括图片内容。|"
            "请分析图片中的场景。"
        ),
    ).split("|")
    if q.strip()
]


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


@events.init.add_listener
def validate_environment(environment: Any, **_: Any) -> None:
    if not IMAGE_PATH.is_file():
        raise RuntimeError(
            f"LOCUST_IMAGE_PATH does not exist: {IMAGE_PATH}. "
            "Set LOCUST_IMAGE_PATH to a valid test image."
        )


class MiniLLaVAFastAPIUser(HttpUser):
    wait_time = between(
        float(os.getenv("LOCUST_WAIT_MIN_S", "0.1")),
        float(os.getenv("LOCUST_WAIT_MAX_S", "1.0")),
    )

    @task(HEALTH_WEIGHT)
    def health(self) -> None:
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status={response.status_code}: {response.text[:300]}")
                return
            try:
                if response.json().get("status") != "ok":
                    response.failure(f"unexpected health body: {response.text[:300]}")
            except ValueError as exc:
                response.failure(f"invalid json: {exc}")

    @task(CHAT_WEIGHT)
    def chat(self) -> None:
        question = random.choice(QUESTIONS) if QUESTIONS else QUESTION
        data = {
            "question": question,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }
        with IMAGE_PATH.open("rb") as image_file:
            files = {
                "image": (
                    IMAGE_PATH.name,
                    image_file,
                    _mime_type(IMAGE_PATH),
                )
            }
            with self.client.post(
                "/chat",
                data=data,
                files=files,
                name="POST /chat",
                timeout=REQUEST_TIMEOUT_S,
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"unexpected status={response.status_code}: {response.text[:500]}")
                    return

                try:
                    body = response.json()
                except ValueError as exc:
                    response.failure(f"invalid json: {exc}")
                    return

                if not body.get("answer"):
                    response.failure(f"missing answer: {body}")
                    return

                timings = body.get("timings", {})
                if "total_ms" not in timings:
                    response.failure(f"missing timings.total_ms: {body}")

