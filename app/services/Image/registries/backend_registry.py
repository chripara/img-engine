from typing import Callable, TypedDict, Type
from utils.enums.checkpoint import Checkpoint
from app.services.Image.backends.base_backend import BaseBackend
from app.services.Image.backends.sdxl_backend import SDXLBackend
from utils.image_converter import ImageConverter

class BackendEntry(TypedDict):
    backend: Type[BaseBackend]
    converter: Callable[..., bytes]

_SDXL_BACKEND: BackendEntry = {
    "backend": SDXLBackend,
    "converter": ImageConverter.Pil_Image_to_Bytes_Png
}

_BACKENDS: dict[Checkpoint, BackendEntry] = {
    Checkpoint.SDXL_BASE: _SDXL_BACKEND,
    Checkpoint.ALBEDO_BASE: _SDXL_BACKEND,
    Checkpoint.JUGGERNAUT_XL: _SDXL_BACKEND,
    Checkpoint.DREAMSHAPER_XL: _SDXL_BACKEND
}