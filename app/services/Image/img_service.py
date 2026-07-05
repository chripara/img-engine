from app.services.image.engine.image_engine import ImageEngine
from app.schemas.generate import GenerateRequest
from PIL import Image

def generate_image(req: GenerateRequest, seeds: list[int] | None = None) -> list[Image.Image]:
    with ImageEngine(req) as engine:
        images = [engine.generate_image(req, seed) for seed in seeds]

    return images

