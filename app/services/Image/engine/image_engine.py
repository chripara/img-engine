from typing import Callable
from flask import current_app
from PIL import Image
from app.schemas.generate import GenerateRequest
from utils.enums import ImgBackend, Checkpoint, Profile
from app.services.image.registries.backend_registry import _BACKENDS, BackendEntry
from app.services.registries.profile_registry import _PROFILES, ProfileSpec
from app.services.image.backends.base_backend import BaseBackend
import gc, torch, random


class ImageEngine:
    def __init__(self, req: GenerateRequest):
        self.profile = req.profile
        self.model = _PROFILES[req.profile].model        
        self.backend = self._get_backend(req)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.model = None
        self.backend.unload()
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def _get_backend(self, req: GenerateRequest) -> BaseBackend: 
        match self.model:
            case (
                Checkpoint.SDXL_BASE |
                Checkpoint.ALBEDO_BASE |
                Checkpoint.JUGGERNAUT_XL | 
                Checkpoint.DREAMSHAPER_XL
            ):
                return _BACKENDS[self.model]["backend"](profile=req.profile)
                
    def generate_image(self, req: GenerateRequest) -> Image.Image:
        if not req.prompt:
            raise ValueError("prompt is required")

        if not req.profile:
            raise ValueError("profile is required")

        print("Prompt:", req.prompt)
        seed = random.randint(req.seed - req.spread, req.seed + req.spread) if req.seed is not None and req.spread is not None else req.seed
        result = self.backend.generate(req.prompt, seed)
        return  result



