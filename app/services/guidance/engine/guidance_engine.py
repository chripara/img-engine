from PIL import Image
from app.schemas.generate import GuidanceSettings, GenerateRequest, GuidanceResult
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.registries.backend_registry import _GUIDANCE_BACKEND
from app.services.registries.profile_registry import _PROFILES
import torch, gc

class GuidanceEngine:
    def __init__(self, req: GenerateRequest):
        self._backend: BaseGuidanceBackend = _GUIDANCE_BACKEND[_PROFILES[req.profile].model]()   # χωρίς ["backend"]

    def __enter__(self) -> "GuidanceEngine":
        return self

    def __exit__(self, *args):
        self._backend.unload()
        del self._backend
        torch.cuda.empty_cache()
        gc.collect()

    def prepare(self, control: GuidanceSettings, image: Image.Image) -> GuidanceResult | None:
        self._backend.load(control.type)
        processed = self._backend.preprocess(image)
        self._backend.unload()

        if processed is None:
            return None

        return GuidanceResult(
            type=control.type,
            image=processed,
            strength=control.strength,
        )