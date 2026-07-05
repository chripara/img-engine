from PIL import Image
from app.schemas.generate import GuidanceSettings
from app.services.guidance.backends.base_guidance_backend import BaseGuidanceBackend
from app.services.guidance.registries.guidance_registry import _SDXL_PREPROCESSORS

class SDXLGuidanceBackend(BaseGuidanceBackend):
    def preprocess(self, image: Image.Image, control: GuidanceSettings) -> Image.Image | None:
        fn = _SDXL_PREPROCESSORS.get(control.type)
        if fn is None:
            return None
        return fn(self, image)