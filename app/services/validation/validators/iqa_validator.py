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
    metric = _load_iqa()
    arr = np.array(image.convert("RGB")).transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        return metric(tensor).item() / 100

def iqa_validator(image: Image.Image) -> GateResult:
    score = _compute_iqa_score(image)
    status = resolve_gate_status(GateType.IQA, score)

    return GateResult(
        gate=GateType.IQA,
        score=score,
        passed=status == GateStatus.PASS,
        suggested=_GATE_MESSAGES[GateType.IQA][status],
    )
