"""Local smoke test that does not require a running vLLM server."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from week19_end2end_service.service.app import create_app
from week19_end2end_service.service.config import Settings


class FakeBackend:
    async def generate(self, *, messages, max_tokens, temperature):
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "image_url"
        assert messages[0]["content"][1]["type"] == "text"
        return f"fake answer; max_tokens={max_tokens}; temperature={temperature}"

    async def close(self):
        return None


def make_png_bytes() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGB", (32, 24), color=(30, 120, 80))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    app = create_app(Settings(rate_limit_per_min=0), backend=FakeBackend())
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            data={"question": "这是什么颜色？", "max_tokens": "16", "temperature": "0"},
            files={"image": ("demo.png", make_png_bytes(), "image/png")},
        )
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    main()

