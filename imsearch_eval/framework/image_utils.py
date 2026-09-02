"""Shared PIL image normalization for caption LLMs, CLIP, and DLQ reload."""

import logging
import os
from io import BytesIO
from typing import Optional

from PIL import Image

_DEFAULT_LLM_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_DEFAULT_LLM_MAX_IMAGE_SIDE = 6144


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes")


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        logging.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value <= 0:
        logging.warning("Invalid %s=%d (must be > 0); using default %d", name, value, default)
        return default
    return value


def llm_image_byte_limiting_enabled() -> bool:
    """When true, apply ``LLM_MAX_IMAGE_SIDE``, byte downscale, and JPEG quality caps."""
    return _env_bool("LLM_IMAGE_BYTE_LIMITING", False)


def llm_max_image_bytes() -> int:
    """Max caption image payload bytes (``LLM_MAX_IMAGE_BYTES``)."""
    return _env_positive_int("LLM_MAX_IMAGE_BYTES", _DEFAULT_LLM_MAX_IMAGE_BYTES)


def llm_max_image_side() -> int:
    """Longest-side pixel cap when byte limiting is enabled (``LLM_MAX_IMAGE_SIDE``)."""
    return _env_positive_int("LLM_MAX_IMAGE_SIDE", _DEFAULT_LLM_MAX_IMAGE_SIDE)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB PIL image (drops alpha / converts palette modes)."""
    if image.mode == "RGB":
        return image
    return image.convert("RGB")


def _cap_longest_side(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.LANCZOS,
    )


def _downscale_until_rgb_under_bytes(image: Image.Image, max_bytes: int) -> Image.Image:
    """Shrink a PIL RGB image until uncompressed payload is under ``max_bytes``."""
    width, height = image.size
    while width * height * 3 > max_bytes:
        width = max(1, int(width * 0.85))
        height = max(1, int(height * 0.85))
        image = image.resize((width, height), Image.LANCZOS)
    return image


def prepare_llm_image(
    image: Image.Image,
    *,
    max_side: Optional[int] = None,
    max_bytes: Optional[int] = None,
    byte_limiting_enabled: Optional[bool] = None,
) -> Image.Image:
    """
    Prepare a PIL image for caption LLMs (Triton UINT8 tensors, etc.).

    Always converts to RGB. When ``LLM_IMAGE_BYTE_LIMITING`` is enabled, also
    caps longest side and downscales to stay under ``LLM_MAX_IMAGE_BYTES``.
    """
    if byte_limiting_enabled is None:
        byte_limiting_enabled = llm_image_byte_limiting_enabled()

    rgb = ensure_rgb(image)
    if not byte_limiting_enabled:
        return rgb

    if max_side is None:
        max_side = llm_max_image_side()
    if max_bytes is None:
        max_bytes = llm_max_image_bytes()

    rgb = _cap_longest_side(rgb, max_side)
    return _downscale_until_rgb_under_bytes(rgb, max_bytes)


def prepare_llm_image_bytes(
    image: Image.Image,
    *,
    max_side: Optional[int] = None,
    jpeg_quality: int = 85,
    max_bytes: Optional[int] = None,
    byte_limiting_enabled: Optional[bool] = None,
) -> tuple[bytes, str]:
    """
    Encode an image for HTTP/OpenAI-style multimodal caption requests.

    Uses JPEG (not PNG). Byte limiting (side cap, downscale, quality stepping)
    follows ``LLM_IMAGE_BYTE_LIMITING``.
    """
    if byte_limiting_enabled is None:
        byte_limiting_enabled = llm_image_byte_limiting_enabled()
    if max_bytes is None:
        max_bytes = llm_max_image_bytes()

    rgb = prepare_llm_image(
        image,
        max_side=max_side,
        max_bytes=max_bytes,
        byte_limiting_enabled=byte_limiting_enabled,
    )

    if not byte_limiting_enabled:
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        return buffer.getvalue(), "jpeg"

    quality = jpeg_quality
    while quality >= 50:
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= max_bytes:
            return data, "jpeg"
        quality -= 10

    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=50, optimize=True)
    return buffer.getvalue(), "jpeg"
