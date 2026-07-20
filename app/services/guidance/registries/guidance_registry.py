from typing import Callable

from utils.enums.guidance import GuidanceType
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector, HEDdetector

_SDXL_PREPROCESSORS: dict[GuidanceType, Callable] = {
    GuidanceType.CANNY:    lambda: CannyDetector(),
    GuidanceType.DEPTH:    lambda: MidasDetector.from_pretrained("lllyasviel/Annotators"),
    GuidanceType.POSE:     lambda: OpenposeDetector.from_pretrained("lllyasviel/ControlNet"),
    GuidanceType.SCRIBBLE: lambda: HEDdetector.from_pretrained("lllyasviel/Annotators"),
}