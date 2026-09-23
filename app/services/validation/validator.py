from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from app.schemas.generate import GateResult
from app.services.validation.validators.tiling_validator import tiling_validator
from app.services.validation.validators.face_validator import face_validator
from app.services.validation.validators.iqa_validator import iqa_validator
from app.services.validation.validators.clip_validator import clip_validator
from app.services.validation.validators.hands_validator import hands_validator
from utils.enums.gate import GateType


def validate(image: Image.Image, prompt: str) -> list[GateResult]:
    with ThreadPoolExecutor() as executor:
        submissions = [
            (GateType.TILING, executor.submit(tiling_validator, image)),
            (GateType.CLIP, executor.submit(clip_validator, image, prompt)),
            (GateType.HANDS, executor.submit(hands_validator, image)),
            (GateType.FACE, executor.submit(face_validator, image)),
            (GateType.IQA, executor.submit(iqa_validator, image)),
        ]

        results: list[GateResult] = []
        for gate, future in submissions:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(GateResult(gate=gate, passed=None, suggested=f"gate error: {e}"))
        return results
