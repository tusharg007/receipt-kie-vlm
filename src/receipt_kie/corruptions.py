"""Deterministic pilot image corruptions for optional robustness evaluation."""

from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter


def gaussian_blur(image: Image.Image, radius: float = 1.2) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg_compression(image: Image.Image, quality: int = 45) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


def reduced_brightness(image: Image.Image, factor: float = 0.65) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def small_rotation(image: Image.Image, seed: int = 42, degrees: float = 3.0) -> Image.Image:
    angle = random.Random(seed).uniform(-degrees, degrees)
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
