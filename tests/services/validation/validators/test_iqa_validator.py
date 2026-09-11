from unittest.mock import patch
import pytest
from PIL import Image

from app.services.validation.validators import iqa_validator as module
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS
from utils.enums.gate import GateType, GateStatus


@pytest.mark.parametrize(
    "score, expected_status",
    [
        (_GATE_THRESHOLDS[GateType.IQA][GateStatus.WARNING] + 0.1, GateStatus.PASS),
        ((_GATE_THRESHOLDS[GateType.IQA][GateStatus.FAIL] + _GATE_THRESHOLDS[GateType.IQA][GateStatus.WARNING]) / 2, GateStatus.WARNING),
        (_GATE_THRESHOLDS[GateType.IQA][GateStatus.FAIL] - 0.05, GateStatus.FAIL),
    ],
)
def test_maps_score_to_expected_status(score, expected_status):
    with patch.object(module, "_compute_iqa_score", return_value=score):
        result = module.iqa_validator(Image.new("RGB", (2, 2)))

    assert result.gate == GateType.IQA
    assert result.score == score
    assert result.passed == (expected_status == GateStatus.PASS)
