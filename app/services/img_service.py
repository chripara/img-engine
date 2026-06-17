import base64
from typing import Callable

from flask import current_app, g
from PIL import Image

from utils.enums import ImgBackend
from app.services.backends.backend_registry import _BACKENDS
from app.services.backends.base_backend import BaseBackend


def generate_image(prompt: str) -> bytes:
    if not prompt:
        raise ValueError("prompt is required")

    backend = _resolve_backend()
    result = get_backend(backend).generate(prompt)
    image_bytes = get_converter(backend)(result) if isinstance(result, Image.Image) else result
    return image_bytes


def get_backend(backend: ImgBackend | None = None) -> BaseBackend:
    backend = backend or _resolve_backend()
    cache_attr = f"backend_{backend.value}"
    if not hasattr(g, cache_attr):
        setattr(g, cache_attr, _BACKENDS[backend]["backend"]())
    return getattr(g, cache_attr)


def get_converter(backend: ImgBackend | None = None) -> Callable[..., bytes]:
    backend = backend or _resolve_backend()
    return _BACKENDS[backend]["converter"]


def _resolve_backend() -> ImgBackend:
    backend = current_app.config["IMG_BACKEND"]
    if backend not in _BACKENDS:
        raise ValueError(f"Invalid backend: {backend}")
    return backend
