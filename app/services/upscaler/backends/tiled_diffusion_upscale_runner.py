from abc import ABC, abstractmethod
from PIL import Image


class TiledDiffusionUpscaleRunner(ABC):

    @abstractmethod
    def load(self, model_path: str) -> None:
        pass

    @abstractmethod
    def run_tile(self, prompt: str, tile: Image.Image, noise_level: int, steps: int) -> Image.Image:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass