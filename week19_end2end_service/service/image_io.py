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
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="image file is empty")

    max_bytes = max_image_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image is too large: {len(raw)} bytes > {max_bytes} bytes",
        )
    image_uuid = hashlib.sha256(raw).hexdigest()

    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image") from exc

    pil_image = pil_image.convert("RGB")
    if max(pil_image.size) > max_image_side:
        pil_image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)

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

