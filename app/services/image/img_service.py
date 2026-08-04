from app.services.image.engine.image_engine import ImageEngine
from app.schemas.generate import GenerateRequest, GuidanceResult
from PIL import Image

def generate_image(req: GenerateRequest, seeds: list[int] | None = None, controls: list[GuidanceResult] | None = None) -> list[Image.Image]:
    guidance_types = [ctr.type for ctr in controls if controls is not None]

    with ImageEngine(req, guidance_types) as engine:
        images = [engine.generate_image(req, seed, controls, index) for index, seed in enumerate(seeds)]

    return images
