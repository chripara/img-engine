from typing import Callable, TypedDict, Type

from utils.enums import ImgBackend
from app.services.backends.base_backend import BaseBackend
from app.services.backends.sdxl_backend import SDXLBackend
from utils.image_converter import ImageConverter


class BackendEntry(TypedDict):
    backend: Type[BaseBackend]
    converter: Callable[..., bytes]

_BACKENDS: dict[ImgBackend, BackendEntry] = {
    ImgBackend.SDXL: {
        "backend": SDXLBackend,
        "converter": ImageConverter.Pil_Image_to_Bytes_Png,
    }
}