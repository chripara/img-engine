from abc import ABC, abstractmethod
from PIL import Image
from app.schemas.generate import GenerateRequest
from utils.enums import GuidanceType


class BaseGuidanceBackend(ABC):
    @abstractmethod
    def preprocess(self, guidanceType: GuidanceType, image: Image.Image) -> Image.Image:
        pass