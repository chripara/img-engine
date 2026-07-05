from typing import Callable

from utils.enums import GuidanceType
from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector, HEDdetector

_SDXL_PREPROCESSORS: dict[GuidanceType, Callable] = {
    GuidanceType.CANNY:    lambda self, img: CannyDetector()(img),
    GuidanceType.DEPTH:    lambda self, img: MidasDetector.from_pretrained("lllyasviel/Annotators")(img),
    GuidanceType.POSE:     lambda self, img: OpenposeDetector.from_pretrained("lllyasviel/ControlNet")(img),
    GuidanceType.SCRIBBLE: lambda self, img: HEDdetector.from_pretrained("lllyasviel/Annotators")(img, scribble=True),
}