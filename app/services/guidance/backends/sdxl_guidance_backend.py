from PIL import Image
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.backends.preprocessor_runner import GuidancePreprocessorRunner
from app.services.guidance.backends.sdxl_preprocessor_runner import SDXLPreprocessorRunner
from utils.enums.guidance import GuidanceType


class SDXLGuidanceBackend(BaseGuidanceBackend):
    def __init__(self, runner: GuidancePreprocessorRunner | None = None) -> None:
        self._runner: GuidancePreprocessorRunner = runner if runner is not None else SDXLPreprocessorRunner()
        self._loaded = False

    def load(self, guidance_type: GuidanceType) -> None:
        self._runner.load(guidance_type)
        self._loaded = True

    def unload(self):
        self._runner.unload()
        self._loaded = False

    def preprocess(self, image: Image.Image) -> Image.Image | None:
        if not self._loaded:
            return None
        return self._runner.run(image)