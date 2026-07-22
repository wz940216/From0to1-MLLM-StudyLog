"""Image validation and encoding helpers."""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class PreparedImage:
    data_url: str
    uuid: str
    width: int
    height: int
    mime_type: str
    bytes_size: int


async def prepare_upload_image(
    image: UploadFile,
    *,
    max_image_mb: int,
    max_image_side: int,
) -> PreparedImage:
    # FastAPI 的 UploadFile 是流式文件对象，先读出 bytes 才能做大小校验和 PIL 解码。
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="image file is empty")

    # 先限制原始上传大小，避免超大文件占满内存或拖慢图片解码。
    max_bytes = max_image_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image is too large: {len(raw)} bytes > {max_bytes} bytes",
        )
    # 用原始图片 bytes 计算稳定 uuid；同一张图重复请求时 vLLM 可以用它做多模态缓存标识。
    image_uuid = hashlib.sha256(raw).hexdigest()

    try:
        # PIL 解码既能确认文件确实是图片，也能统一后续的颜色空间和尺寸处理。
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image") from exc

    pil_image = pil_image.convert("RGB")
    if max(pil_image.size) > max_image_side:
        # 控制最长边，减少上传大图带来的预处理和 vLLM 图像处理开销。
        pil_image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)

    # vLLM OpenAI 接口可以接收 data URL；这里把处理后的图片编码进 JSON 请求体。
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    width, height = pil_image.size
    return PreparedImage(
        data_url=f"data:image/jpeg;base64,{encoded}",
        uuid=image_uuid,
        width=width,
        height=height,
        mime_type="image/jpeg",
        bytes_size=len(raw),
    )

