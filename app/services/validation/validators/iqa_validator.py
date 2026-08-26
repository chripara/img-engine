from PIL import Image
from app.schemas.generate import GateResult
from utils.enums.gate import GateType, GateStatus
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, _GATE_MESSAGES
import torch, pyiqa
import numpy as np

_iqa_metric = None

def _load_iqa():
    global _iqa_metric
    if _iqa_metric is None:
        _iqa_metric = pyiqa.create_metric("musiq")
    return _iqa_metric

def iqa_validator(image: Image.Image) -> GateResult:
    metric = _load_iqa()
    arr = np.array(image.convert("RGB")).transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        score = metric(tensor).item()/100

    status = GateStatus.FAIL \
        if score < _GATE_THRESHOLDS[GateType.IQA][GateStatus.FAIL] \
        else GateStatus.WARNING \
        if score < _GATE_THRESHOLDS[GateType.IQA][GateStatus.WARNING] \
        else GateStatus.PASS

    return GateResult(
        gate=GateType.IQA,
        score=score,
        passed=status == GateStatus.PASS,
        suggested=_GATE_MESSAGES[GateType.IQA][status],
    )