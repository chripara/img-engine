from dataclasses import dataclass

from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.backends.sdxl_guidance_backend import SDXLGuidanceBackend
from utils.enums import Checkpoint, GuidanceType


@dataclass
class GuidanceDetails:
    backend: type[BaseGuidanceBackend]      # πάντα SDXLGuidanceBackend για SDXL
    defaults: dict[GuidanceType, float]

_GUIDANCE_DETAILS: dict[Checkpoint, GuidanceDetails] = {
    Checkpoint.SDXL_BASE: GuidanceDetails(
        backend=SDXLGuidanceBackend,
        defaults={
            GuidanceType.CANNY:    0.70,
            GuidanceType.DEPTH:    0.60,
            GuidanceType.POSE:     0.75,
            GuidanceType.SCRIBBLE: 0.80,
        },
    ),
    Checkpoint.ALBEDO_BASE: GuidanceDetails(
        backend=SDXLGuidanceBackend,
        defaults={
            GuidanceType.CANNY:    0.65,
            GuidanceType.DEPTH:    0.55,
            GuidanceType.POSE:     0.70,
            GuidanceType.SCRIBBLE: 0.75,
        },
    ),
    Checkpoint.JUGGERNAUT_XL: GuidanceDetails(
        backend=SDXLGuidanceBackend,
        defaults={
            GuidanceType.CANNY:    0.65,
            GuidanceType.DEPTH:    0.55,
            GuidanceType.POSE:     0.70,
            GuidanceType.SCRIBBLE: 0.75,
        },
    ),
    Checkpoint.DREAMSHAPER_XL: GuidanceDetails(
        backend=SDXLGuidanceBackend,
        defaults={
            GuidanceType.CANNY:    0.85,
            GuidanceType.DEPTH:    0.65,
            GuidanceType.POSE:     0.80,
            GuidanceType.SCRIBBLE: 0.90,
        },
    ),
}

_SDXL_CONTROLNET_MODELS: dict[GuidanceType, str] = {
    GuidanceType.CANNY:    "diffusers/controlnet-canny-sdxl-1.0",
    GuidanceType.DEPTH:    "diffusers/controlnet-depth-sdxl-1.0",
    GuidanceType.POSE:     "thibaud/controlnet-openpose-sdxl-1.0",
    GuidanceType.SCRIBBLE: "xinsir/controlnet-scribble-sdxl-1.0",
}