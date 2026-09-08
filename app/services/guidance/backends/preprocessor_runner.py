from abc import ABC, abstractmethod
from PIL import Image
from utils.enums.guidance import GuidanceType


class GuidancePreprocessorRunner(ABC):
    @abstractmethod
    def load(self, guidance_type: GuidanceType) -> None:
        pass

    @abstractmethod
    def run(self, image: Image.Image) -> Image.Image:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass