from PIL import Image
from app.schemas.generate import GuidanceSettings
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.backends.sdxl_guidance_backend import SDXLGuidanceBackend

class GuidanceEngine(BaseGuidanceBackend):
    def __init__(self, backend: BaseGuidanceBackend):
        self._backend = backend

    def prepare(self, control: GuidanceSettings, image: Image.Image) -> Image.Image | None:
        return self._backend.preprocess(image, control)

