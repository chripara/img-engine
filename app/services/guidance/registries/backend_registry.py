from PIL import Image

from app.services.guidance.backends.sdxl_guidance_backend import SDXLGuidanceBackend
from utils.enums import Checkpoint, GuidanceType
from typing import Callable, TypedDict, Type
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.registries.guidance_registry import _SDXL_PREPROCESSORS

class GuidanceBackendEntry(TypedDict):
    backend: Type[BaseGuidanceBackend]
    preprocessor: Callable[[Image.Image], Image.Image]

_SDXL_BACKEND: list[GuidanceBackendEntry] = [
    {
        "backend": SDXLGuidanceBackend,
        "preprocessor": _SDXL_PREPROCESSORS[GuidanceType.CANNY]
    },
    {
        "backend": SDXLGuidanceBackend,
        "preprocessor": _SDXL_PREPROCESSORS[GuidanceType.POSE]
    },
    {
        "backend": SDXLGuidanceBackend,
        "preprocessor": _SDXL_PREPROCESSORS[GuidanceType.DEPTH]
    },
    {
        "backend": SDXLGuidanceBackend,
        "preprocessor": _SDXL_PREPROCESSORS[GuidanceType.SCRIBBLE]
    }
]

_GUIDANCE_BACKEND: dict[Checkpoint,list[GuidanceBackendEntry] ] = {
    Checkpoint.ALBEDO_BASE: _SDXL_BACKEND,
    Checkpoint.JUGGERNAUT_XL: _SDXL_BACKEND,
    Checkpoint.DREAMSHAPER_XL: _SDXL_BACKEND
}