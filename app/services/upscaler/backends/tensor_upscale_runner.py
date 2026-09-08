from abc import ABC, abstractmethod
from PIL import Image


class TensorUpscaleRunner(ABC):

    @abstractmethod
    def load(self, model_path: str) -> None:
        pass

    @abstractmethod
    def run(self, image: Image.Image) -> Image.Image:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass