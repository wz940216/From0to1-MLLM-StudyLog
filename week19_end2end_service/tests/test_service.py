from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from week19_end2end_service.service.app import create_app
from week19_end2end_service.service.config import Settings


class FakeBackend:
    def __init__(self):
        self.calls = []

    async def generate(self, *, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        return "这是测试回答。"

    async def close(self):
        return None


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), color=(200, 40, 50)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_chat_pipeline_returns_answer_and_timings():
    backend = FakeBackend()
    app = create_app(Settings(rate_limit_per_min=0), backend=backend)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            data={"question": "描述图片", "max_tokens": "32", "temperature": "0"},
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "这是测试回答。"
    assert data["image"]["width"] == 48
    assert data["image"]["height"] == 32
    assert set(data["timings"]) == {"preprocess_ms", "prompt_ms", "generate_ms", "total_ms"}
    messages, max_tokens, temperature = backend.calls[0]
    assert max_tokens == 32
    assert temperature == 0.0
    assert messages[0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert isinstance(messages[0]["content"][0]["uuid"], str)
    assert len(messages[0]["content"][0]["uuid"]) == 64
    assert messages[0]["content"][1]["text"] == "描述图片"


def test_rate_limit_rejects_second_request_in_window():
    app = create_app(Settings(rate_limit_per_min=1), backend=FakeBackend())

    with TestClient(app) as client:
        first = client.post(
            "/chat",
            data={"question": "one"},
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        second = client.post(
            "/chat",
            data={"question": "two"},
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert first.status_code == 200
    assert second.status_code == 429

