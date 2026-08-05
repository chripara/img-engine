from app.services.upscaler.engine.upscaler_engine import UpscalerEngine
from app.schemas.generate import GenerateRequest
from app.services.registries.profile_registry import ProfileSpec
from PIL import Image

def upscale_image(
    req: GenerateRequest, 
    spec: ProfileSpec, 
    imgs: list[Image.Image],
    seeds: list[int],
) -> list[Image.Image]:
    with UpscalerEngine(req, spec) as engine:
        images = [engine.upscale_image(img, req, index, seeds[index]) for index ,img in enumerate(imgs)]
    return images