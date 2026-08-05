from abc import ABC, abstractmethod
from PIL import Image
from app.schemas.generate import GenerateRequest
from typing import Self

class BaseBackend(ABC):
    @abstractmethod
    def upscale(self, image: Image.Image, req: GenerateRequest, index: int = 0, seed: int | None = None) -> Image.Image:
        pass

    def __enter__(self) -> Self:
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()