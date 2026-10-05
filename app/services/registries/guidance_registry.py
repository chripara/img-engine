from dataclasses import dataclass
from typing import Callable

from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector, HEDdetector
from diffusers import ControlNetModel, SD3ControlNetModel, FluxControlNetModel

from utils.enums.checkpoint import Checkpoint
from utils.enums.guidance import GuidanceType
from utils.enums.profile import Profile

# Single source of truth for everything guidance-related (ControlNet preprocessing and models).
# This module is a leaf: it must not import any backend or engine, to avoid circular imports.

# Preprocessor (annotator) factory per guidance type; each detector is built lazily on load.
_SDXL_PREPROCESSORS: dict[GuidanceType, Callable] = {
    GuidanceType.CANNY:    lambda: CannyDetector(),
    GuidanceType.DEPTH:    lambda: MidasDetector.from_pretrained("lllyasviel/Annotators"),
    GuidanceType.POSE:     lambda: OpenposeDetector.from_pretrained("lllyasviel/ControlNet"),
    GuidanceType.SCRIBBLE: lambda: HEDdetector.from_pretrained("lllyasviel/Annotators"),
}

# ControlNet repo id per guidance type.
_SDXL_CONTROLNET_MODELS: dict[GuidanceType, str] = {
    GuidanceType.CANNY:    "diffusers/controlnet-canny-sdxl-1.0",
    GuidanceType.DEPTH:    "diffusers/controlnet-depth-sdxl-1.0",
    GuidanceType.POSE:     "thibaud/controlnet-openpose-sdxl-1.0",
    GuidanceType.SCRIBBLE: "xinsir/controlnet-scribble-sdxl-1.0",
}

# ControlNet model class per profile.
_GUIDANCE_MODELS: dict[Profile, type[ControlNetModel | SD3ControlNetModel | FluxControlNetModel]] = {
    Profile.CHARACTER: ControlNetModel,
    Profile.PRODUCT: ControlNetModel,
    Profile.SCENE_FRAME: ControlNetModel,
}

# Default control strength per guidance type, for each checkpoint.
@dataclass
class GuidanceDetails:
    defaults: dict[GuidanceType, float]

_GUIDANCE_DETAILS: dict[Checkpoint, GuidanceDetails] = {
    Checkpoint.SDXL_BASE: GuidanceDetails(
        defaults={
            GuidanceType.CANNY:    0.70,
            GuidanceType.DEPTH:    0.60,
            GuidanceType.POSE:     0.75,
            GuidanceType.SCRIBBLE: 0.80,
        },
    ),
    Checkpoint.ALBEDO_BASE: GuidanceDetails(
        defaults={
            GuidanceType.CANNY:    0.65,
            GuidanceType.DEPTH:    0.55,
            GuidanceType.POSE:     0.70,
            GuidanceType.SCRIBBLE: 0.75,
        },
    ),
    Checkpoint.JUGGERNAUT_XL: GuidanceDetails(
        defaults={
            GuidanceType.CANNY:    0.65,
            GuidanceType.DEPTH:    0.55,
            GuidanceType.POSE:     0.70,
            GuidanceType.SCRIBBLE: 0.75,
        },
    ),
    Checkpoint.DREAMSHAPER_XL: GuidanceDetails(
        defaults={
            GuidanceType.CANNY:    0.85,
            GuidanceType.DEPTH:    0.65,
            GuidanceType.POSE:     0.80,
            GuidanceType.SCRIBBLE: 0.90,
        },
    ),
}
