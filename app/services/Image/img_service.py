from app.services.image.engine.image_engine import ImageEngine
from app.schemas.generate import GenerateRequest
from PIL import Image

def generate_image(req: GenerateRequest) -> list[Image.Image]:
    with ImageEngine(req) as engine:
        images = [engine.generate_image(req) for _ in range(req.num_images)]

    return images

