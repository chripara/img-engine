from PIL import Image
from app.schemas.generate import GateResult
from utils.enums.gate import GateType, GateStatus
from app.services.validation.registries.validator_registry import _GATE_MESSAGES, resolve_gate_status
import torch, pyiqa
import numpy as np

_iqa_metric = None

def _load_iqa():
    global _iqa_metric
    if _iqa_metric is None:
        _iqa_metric = pyiqa.create_metric("musiq")
    return _iqa_metric

def _compute_iqa_score(image: Image.Image) -> float:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("IQA: input image size=%s", image.size)

    metric = _load_iqa()
    arr = np.array(image.convert("RGB")).transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0) / 255.0

    logger.info("IQA: VRAM before metric(): allocated=%.2fGB reserved=%.2fGB",
                torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

    with torch.no_grad():
        result = metric(tensor).item() / 100

    logger.info("IQA: VRAM after metric(): allocated=%.2fGB reserved=%.2fGB",
                torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9)

    return result

def iqa_validator(image: Image.Image) -> GateResult:
    score = _compute_iqa_score(image)
    status = resolve_gate_status(GateType.IQA, score)

    return GateResult(
        gate=GateType.IQA,
        score=score,
        passed=status == GateStatus.PASS,
        suggested=_GATE_MESSAGES[GateType.IQA][status],
    )
