from PIL import Image
import numpy as np
from app.schemas.generate import GateResult
from utils.enums.gate import GateStatus, GateType
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, _GATE_MESSAGES

def tiling_validator(image: Image.Image) -> GateResult:
    arr = np.array(image.convert("L"), dtype=float)
    arr -= arr.mean()

    fft  = np.fft.fft2(arr)
    psd  = np.abs(fft) ** 2
    corr = np.fft.ifft2(psd).real
    corr /= corr.max()

    corr[0:10,  0:10]  = 0
    corr[-10:,  0:10]  = 0
    corr[0:10,  -10:]  = 0
    corr[-10:,  -10:]  = 0

    score = 1.0 - corr.max()

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
