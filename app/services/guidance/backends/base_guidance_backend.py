from abc import ABC, abstractmethod
from PIL import Image
from app.schemas.generate import GenerateRequest
from utils.enums import GuidanceType


class BaseGuidanceBackend(ABC):

    @abstractmethod
    def load(self, guidance_type: GuidanceType) -> None:
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Image.Image:
        pass