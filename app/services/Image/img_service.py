from typing import Callable
from flask import current_app
from PIL import Image
from app.schemas.generate import GenerateRequest
from utils.enums import ImgBackend, Checkpoint, Profile
from app.services.Image.backends.backend_registry import _BACKENDS, BackendEntry
from app.services.Image.backends.profile_registry import _PROFILES, ProfileSpec
from app.services.Image.backends.base_backend import BaseBackend


_instances: dict[Checkpoint, BaseBackend] = {}


def generate_image(req: GenerateRequest) -> bytes:
    if not req.prompt:
        raise ValueError("prompt is required")

    if not req.profile:
        raise ValueError("profile is required")

    print("Prompt:", req.prompt)
    result = get_backend(req.profile).generate(req.prompt)
    image_bytes = get_converter(req.profile)(result) if isinstance(result, Image.Image) else result
    return  image_bytes


def get_backend(selected_profile: Profile | None = None) -> BaseBackend:
    profile = selected_profile if selected_profile else _resolve_profile()
    #cache_attr = f"backend_{backend['backend'].__name__}"
    if  _PROFILES[profile].model not in _instances:
        _instances[ _PROFILES[profile].model] = _BACKENDS[ _PROFILES[profile].model]["backend"](profile=profile)

    return _instances[ _PROFILES[profile].model]


def get_converter(selected_profile: Profile | None = None) -> Callable[..., bytes]:
    profile = _PROFILES[selected_profile] if selected_profile else _PROFILES[_resolve_profile()]
    checkpoint = profile.model  # Assuming Profile has a value attribute

    return _BACKENDS[checkpoint]["converter"]


def _resolve_profile() -> Profile:
    profile = current_app.config["PROFILE"]
    if profile not in Profile:
        raise ValueError(f"Invalid profile: {profile}")
    return profile # _PROFILES[profile]