from typing import Callable
from flask import current_app
from PIL import Image
from app.schemas.generate import GenerateRequest
from utils.enums import ImgBackend, Checkpoint, Profile
from app.services.image.backends.backend_registry import _BACKENDS, BackendEntry
from app.services.image.backends.profile_registry import _PROFILES, ProfileSpec
from app.services.image.backends.base_backend import BaseBackend
import gc, torch

_instances: dict[Checkpoint, BaseBackend] = {}


class ImageEngine:
    def __init__(self, req: GenerateRequest):
        self.profile = req.profile
        self.model = _PROFILES[req.profile].model
        if self.model not in _instances:
            _instances[self.model] = _BACKENDS[self.model]["backend"](profile=req.profile)
        self.backend = _instances[self.model]
        self.converter = _BACKENDS[self.model]["converter"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del _instances[self.model]
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


    def generate_image(self, req: GenerateRequest) -> bytes:
        if not req.prompt:
            raise ValueError("prompt is required")

        if not req.profile:
            raise ValueError("profile is required")

        print("Prompt:", req.prompt)
        result = self.backend.generate(req.prompt, req.seed)
        image_bytes = self._get_converter(req.profile)(result) if isinstance(result, Image.Image) else result
        return  image_bytes


    # def _get_backend(self, selected_profile: Profile | None = None) -> BaseBackend:
    #     profile = selected_profile if selected_profile else self._resolve_profile()
    #     #cache_attr = f"backend_{backend['backend'].__name__}"
    #     if  _PROFILES[profile].model not in _instances:
    #         _instances[ _PROFILES[profile].model] = _BACKENDS[ _PROFILES[profile].model]["backend"](profile=profile)

    #     return _instances[ _PROFILES[profile].model]


    def _get_converter(self, selected_profile: Profile | None = None) -> Callable[..., bytes]:
        profile = _PROFILES[selected_profile] if selected_profile else _PROFILES[self.profile]
        checkpoint = profile.model  # Assuming Profile has a value attribute

        return _BACKENDS[checkpoint]["converter"]


    # def _resolve_profile(self) -> Profile:
    #     profile = current_app.config["PROFILE"]
    #     if profile not in Profile:
    #         raise ValueError(f"Invalid profile: {profile}")
    #     return profile