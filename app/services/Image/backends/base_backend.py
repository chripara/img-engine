from abc import ABC, abstractmethod
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline

from app.services.registries.profile_registry import _PROFILES
from app.schemas.generate import GuidanceResult
from app.services.registries.image_registry import Dimensions
from utils.enums import Profile, GuidanceType
from app.services.image.registries.lora_registry import _LORA_REGISTRY
from app.services.image.registries.guidance_registry import _GUIDANCE_MODELS, _SDXL_CONTROLNET_MODELS
import torch

class BaseBackend(ABC):
    def __init__(self):
        self._pipe: DiffusionPipeline | None = None

    @abstractmethod
    def load(self, profile: Profile, lora_weight: float | None, use_controlnet: bool, guidance_types: list[GuidanceType]) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str | None, dimensions: Dimensions, seed: int | None, controls: list[GuidanceResult] | None) -> Image.Image:
        pass

    def _define_vae(self, profile: Profile):
        self._vae = AutoencoderKL.from_pretrained(
            _PROFILES[profile].vae_id,
            torch_dtype=torch.float16
        ) if _PROFILES[profile].vae_id else None

    def _define_guidance(self, profile: Profile, guidance_types: list[GuidanceType]):
        guiding_model = _GUIDANCE_MODELS[profile]
        self._controlnets = [
            guiding_model.from_pretrained(_SDXL_CONTROLNET_MODELS[gt], torch_dtype=torch.float16)
            for gt in guidance_types
        ]

    def _define_lora(self, profile: Profile, lora_weight: float | None):
        if lora_weight is not None and lora_weight > 0.0:
            self._pipe.load_lora_weights(_LORA_REGISTRY[profile], adapter_name=_LORA_REGISTRY[profile])

            self._pipe.set_adapters(
                _LORA_REGISTRY[profile],
                adapter_weights=lora_weight,
            )

            self._pipe.fuse_lora()

    def _define_scheduler(self, profile: Profile):
        self._pipe.scheduler = _PROFILES[profile].scheduler.from_config(self._pipe.scheduler.config)

