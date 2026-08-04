from abc import ABC, abstractmethod
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline

from app.services.registries.profile_registry import _PROFILES
from app.schemas.generate import GuidanceResult
from app.services.registries.image_registry import Dimensions
from utils.enums.profile import Profile
from utils.enums.guidance import GuidanceType

from app.services.image.registries.stype_presets import _STYLE_PRESET_REGISTRY
from app.services.image.registries.guidance_registry import _GUIDANCE_MODELS, _SDXL_CONTROLNET_MODELS
import torch

from utils.enums.style_presets import StylePreset


class BaseBackend(ABC):
    def __init__(self):
        self._pipe: DiffusionPipeline | None = None

    @abstractmethod
    def load(self, profile: Profile, style_preset: StylePreset, lora_weight: float | None, use_controlnet: bool, guidance_types: list[GuidanceType]) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str | None, dimensions: Dimensions, seed: int | None, controls: list[GuidanceResult] | None, index: int = 0) -> Image.Image:
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

    def _define_lora(self, style_preset: StylePreset | None, lora_weight: float | None):
        if style_preset is not None:
            if lora_weight is None:
                strength=0.8
            else:
                strength=lora_weight
            self._pipe.load_lora_weights(_STYLE_PRESET_REGISTRY[style_preset], adapter_name=style_preset.value)

            self._pipe.set_adapters(
                style_preset.value,
                adapter_weights=strength,
            )

            print("lora stregth: ",strength)
            self._pipe.fuse_lora()

    def _define_scheduler(self, profile: Profile):
        self._pipe.scheduler = _PROFILES[profile].scheduler.from_config(self._pipe.scheduler.config)

