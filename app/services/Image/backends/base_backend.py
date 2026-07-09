from abc import ABC, abstractmethod
from PIL import Image

from app.schemas.generate import GuidanceResult
from utils.enums import Profile, GuidanceType


class BaseBackend(ABC):
    @abstractmethod
    def load(self, profile: Profile, use_controlnet: bool, guidance_types: list[GuidanceType]) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, seed: int | None, controls: list[GuidanceResult]) -> Image.Image:
        pass

