import base64, io
from typing import Callable

from flask import current_app
from PIL import Image

from utils.enums import ImgBackend, Checkpoint, Profile
from app.services.backends.backend_registry import _BACKENDS, BackendEntry
from app.services.backends.profile_registry import _PROFILES, ProfileSpec
from app.services.backends.base_backend import BaseBackend


_instances: dict[Checkpoint, BaseBackend] = {}


def generate_image(prompt: str, selected_profile: Profile) -> bytes:
    if not prompt:
        raise ValueError("prompt is required")

    if not selected_profile:
        raise ValueError("profile is required")

    result = get_backend(selected_profile).generate(prompt)
    image_bytes = get_converter(selected_profile)(result) if isinstance(result, Image.Image) else result
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