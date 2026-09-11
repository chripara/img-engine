from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from app.schemas.generate import GateResult
from app.services.validation.validators.tiling_validator import tiling_validator
from app.services.validation.validators.face_validator import face_validator
from app.services.validation.validators.iqa_validator import iqa_validator
from app.services.validation.validators.clip_validator import clip_validator
from app.services.validation.validators.hands_validator import hands_validator

def validate(image: Image.Image, prompt: str) -> list[GateResult]:
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(tiling_validator, image),
            executor.submit(clip_validator, image, prompt),
            executor.submit(hands_validator, image),
            executor.submit(face_validator, image),
            executor.submit(iqa_validator, image),
        ]

        return [f.result() for f in futures]
