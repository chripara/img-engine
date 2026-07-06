from PIL import Image
from typing import Type
from app.schemas.generate import GuidanceSettings, GenerateRequest, GuidanceResult
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.registries.backend_registry import _GUIDANCE_BACKEND, _SDXL_PREPROCESSORS
from app.services.registries.profile_registry import _PROFILES
import torch, gc

class GuidanceEngine(BaseGuidanceBackend):
    def __init__(self, req: GenerateRequest):
        self._backend: Type[BaseGuidanceBackend] = _GUIDANCE_BACKEND[_SDXL_PREPROCESSORS[_PROFILES[req.profile].model]]

    def __enter__(self) -> GuidanceSettings:
        return self

    def __exit__(self, *args):
        self._backend.unload()
        del self._backend
        torch.cuda.empty_cache()
        gc.collect()

    def prepare(self, control: GuidanceSettings, image: Image.Image) -> GuidanceResult | None:
        guidance_result = GuidanceResult()
        self._backend.load(control.type)
        guidance_result.image = self._backend.preprocess(image)
        guidance_result.strength = control.strength
        self._backend.unload()

        return guidance_result

