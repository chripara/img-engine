from typing import Type, TypedDict

from utils.enums import Checkpoint
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.backends.sdxl_guidance_backend import SDXLGuidanceBackend

_GUIDANCE_BACKEND: dict[Checkpoint, Type[BaseGuidanceBackend]] = {
    Checkpoint.ALBEDO_BASE: SDXLGuidanceBackend,
    Checkpoint.JUGGERNAUT_XL: SDXLGuidanceBackend,
    Checkpoint.DREAMSHAPER_XL: SDXLGuidanceBackend,
}