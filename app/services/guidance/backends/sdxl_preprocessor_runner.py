from typing import Callable
from PIL import Image
import torch, gc

from app.services.guidance.backends.preprocessor_runner import GuidancePreprocessorRunner
from app.services.guidance.registries.guidance_registry import _SDXL_PREPROCESSORS
from utils.enums.guidance import GuidanceType


class SDXLPreprocessorRunner(GuidancePreprocessorRunner):
    def __init__(self) -> None:
        self._detector: Callable | None = None

    def load(self, guidance_type: GuidanceType) -> None:
        self._detector = _SDXL_PREPROCESSORS[guidance_type]()

    def run(self, image: Image.Image) -> Image.Image:
        return self._detector(image)

    def unload(self) -> None:
        self._detector = None
        torch.cuda.empty_cache()
        gc.collect()