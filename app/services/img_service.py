import base64, io
from typing import Callable

from flask import current_app
from PIL import Image

from utils.enums import ImgBackend, Checkpoint
from app.services.backends.backend_registry import _BACKENDS, BackendEntry
from app.services.backends.base_backend import BaseBackend


_instances: dict[Checkpoint, BaseBackend] = {}


def generate_image(prompt: str, selected_checkpoint: Checkpoint) -> bytes:
    if not prompt:
        raise ValueError("prompt is required")
    
    if not selected_checkpoint:
        raise ValueError("backend is required")

    result = get_backend(selected_checkpoint).generate(prompt)
    image_bytes = get_converter(selected_checkpoint)(result) if isinstance(result, Image.Image) else result
    return  image_bytes


def get_backend(checkpoint: Checkpoint | None = None) -> BaseBackend:    
    checkpoint = checkpoint or _resolve_checkpoint()
    backend = _BACKENDS[checkpoint]
    cache_attr = f"backend_{backend['backend'].__name__}"
    if checkpoint not in _instances:
        _instances[checkpoint] = _BACKENDS[checkpoint]["backend"](checkpoint=checkpoint)
    
    # if not hasattr(g, cache_attr):
    #     setattr(g, cache_attr, _BACKENDS[checkpoint]["backend"]())
    
    return _instances[checkpoint]



def get_converter(checkpoint: Checkpoint | None = None) -> Callable[..., bytes]:
    checkpoint = checkpoint or _resolve_checkpoint()
    
    return _BACKENDS[checkpoint]["converter"]


def _resolve_checkpoint() -> Checkpoint:
    checkpoint = current_app.config["CHECKPOINT"]
    if checkpoint not in _BACKENDS:
        raise ValueError(f"Invalid checkpoint: {checkpoint}")
    return checkpoint
