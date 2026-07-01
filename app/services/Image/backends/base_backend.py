from abc import ABC, abstractmethod
from PIL import Image

class BaseBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, seed: int | None) -> Image.Image:
        pass