from app.services.image.engine.image_engine import ImageEngine
from app.schemas.generate import GenerateRequest

def generate_image(req: GenerateRequest) -> list[bytes]:
    with ImageEngine(req) as engine:
        images = [engine.generate_image(req) for _ in range(req.num_images)]
    return images