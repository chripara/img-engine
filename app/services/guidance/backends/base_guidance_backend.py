from abc import ABC, abstractmethod
from PIL import Image
from utils.enums.guidance import GuidanceType


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