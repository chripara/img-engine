from PIL import Image
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.registries.guidance_registry import _SDXL_PREPROCESSORS
from utils.enums.guidance import GuidanceType
from typing import Callable

import torch, gc

class SDXLGuidanceBackend(BaseGuidanceBackend):

    def __init__(self):
        self._detector: Callable | None = None

    def load(self, guidance_type: GuidanceType) -> None:
        self._detector = _SDXL_PREPROCESSORS[guidance_type]()

    def unload(self):
        self._detector = None
        torch.cuda.empty_cache()
        gc.collect()

    def preprocess(self, image: Image.Image) -> Image.Image | None:
        if self._detector is None:
            return None

        return self._detector(image)