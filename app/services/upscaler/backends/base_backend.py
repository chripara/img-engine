from abc import ABC, abstractmethod
from PIL import Image
from app.schemas.generate import GenerateRequest

class BaseBackend(ABC):
    @abstractmethod
    def upscale(self, image: Image.Image, req: GenerateRequest) -> Image.Image:
        pass