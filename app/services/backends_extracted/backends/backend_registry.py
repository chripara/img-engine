from typing import Callable, TypedDict, Type

from utils.enums import ImgBackend, Checkpoint, ModelSource
from app.services.backends.base_backend import BaseBackend
from app.services.backends.sdxl_backend import SDXLBackend
from utils.image_converter import ImageConverter


class BackendEntry(TypedDict):
    backend: Type[BaseBackend]
    converter: Callable[..., bytes]

_BACKENDS: dict[Checkpoint, BackendEntry] = {
    Checkpoint.SDXL_BASE: {
        "backend": SDXLBackend,
        "converter": ImageConverter.Pil_Image_to_Bytes_Png,
    },
    Checkpoint.ALBEDO_BASE: {
        "backend": SDXLBackend,
        "converter": ImageConverter.Pil_Image_to_Bytes_Png,
    },
    Checkpoint.JUGGERNAUT_XL: {
        "backend": SDXLBackend,
        "converter": ImageConverter.Pil_Image_to_Bytes_Png,
    },
    Checkpoint.DREAMSHAPER_XL: {
        "backend": SDXLBackend,
        "converter": ImageConverter.Pil_Image_to_Bytes_Png,
    }
}