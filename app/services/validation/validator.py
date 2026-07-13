from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, _GATE_MESSAGES
from app.schemas.generate import GateResult
from utils.enums import GateType, GateStatus


def validate(image: Image.Image) -> list[GateResult]:
    gate_results: list[GateResult] = []

    gate_results.append(_tiling_validator(image))

    return gate_results

def _tiling_validator(image: Image.Image) -> GateResult:
    arr = np.array(image.convert("L"))
    h, w = arr.shape
    hh, hw = h // 2, w // 2

    tl = arr[:hh, :hw]
    tr = arr[:hh, hw:]
    bl = arr[hh:, :hw]
    br = arr[hh:, hw:]

    scores = [
        ssim(tl, tr),
        ssim(tl, bl),
        ssim(tl, br),
        ssim(tr, bl),
        ssim(tr, br),
        ssim(bl, br),
    ]

    score = 1.0 - max(scores)

    status = GateStatus.FAIL \
        if score < _GATE_THRESHOLDS[GateType.TILING][GateStatus.FAIL] \
        else GateStatus.WARNING \
        if score < _GATE_THRESHOLDS[GateType.TILING][GateStatus.WARNING] \
        else GateStatus.PASS

    return GateResult(
        gate = GateType.TILING,
        score = score,
        passed  = status == GateStatus.PASS,
        suggested = _GATE_MESSAGES[GateType.TILING][status],
    )