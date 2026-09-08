from abc import ABC, abstractmethod
from PIL import Image

from app.schemas.generate import GuidanceResult
from app.services.registries.image_registry import Dimensions
from utils.enums.profile import Profile
from utils.enums.guidance import GuidanceType
from utils.enums.style_presets import StylePreset


class BaseBackend(ABC):

    @abstractmethod
    def load(self, profile: Profile, style_preset: StylePreset, lora_weight: float | None, use_controlnet: bool, guidance_types: list[GuidanceType]) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, negative_prompt: str | None, dimensions: Dimensions, seed: int | None, controls: list[GuidanceResult] | None, index: int = 0) -> Image.Image:
        pass