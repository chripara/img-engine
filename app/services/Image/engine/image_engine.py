from PIL import Image
from app.schemas.generate import GenerateRequest, GuidanceResult
from utils.enums.checkpoint import Checkpoint
from utils.enums.guidance import GuidanceType
from utils.enums.aspect_ratio import AspectRatio
from app.services.Image.registries.backend_registry import _BACKENDS, BackendEntry
from app.services.registries.profile_registry import _PROFILES, ProfileSpec
from app.services.registries.image_registry import _ASPECT_RATIOS
from app.services.Image.backends.base_backend import BaseBackend
import gc, torch


class ImageEngine:
    def __init__(self, req: GenerateRequest, guidance_types: list[GuidanceType]):
        self._profile = req.profile
        self._model = _PROFILES[req.profile].model
        self._backend = self._get_backend(req)
        self._guidance_types = guidance_types
        self._style_preset = req.style_preset
        self._lora_weights = req.lora_strength

    def __enter__(self):
        self._backend.load(self._profile, self._style_preset, self._lora_weights, len(self._guidance_types) > 0, self._guidance_types)

        return self

    def __exit__(self, *args):
        self._model = None
        self._backend.unload()
        del self._model
        torch.cuda.empty_cache()
        gc.collect()

    def _get_backend(self, req: GenerateRequest) -> BaseBackend: 
        match self._model:
            case (
                Checkpoint.SDXL_BASE |
                Checkpoint.ALBEDO_BASE |
                Checkpoint.JUGGERNAUT_XL | 
                Checkpoint.DREAMSHAPER_XL
            ):
                return _BACKENDS[self._model]["backend"](profile=req.profile)
                
    def generate_image(self, req: GenerateRequest, seed: int | None = None, controls: list[GuidanceResult] | None = None) -> Image.Image:
        if not req.prompt:
            raise ValueError("prompt is required")

        if not req.profile:
            raise ValueError("profile is required")

        dimensions = _ASPECT_RATIOS[req.aspect_ratio] if req.aspect_ratio else _ASPECT_RATIOS[AspectRatio.SQUARE]

        print("Prompt:", req.prompt)
        result = self._backend.generate(req.prompt, req.negative_prompt, dimensions, seed, controls)

        return  result
