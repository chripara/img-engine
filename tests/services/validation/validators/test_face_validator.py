from unittest.mock import patch
import pytest
from PIL import Image

from app.services.validation.validators import face_validator as module
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS
from utils.enums.gate import GateType, GateStatus


def test_returns_not_applicable_when_no_face_detected():
    with patch.object(module, "_detect_face_score", return_value=None):
        result = module.face_validator(Image.new("RGB", (2, 2)))

    assert result.score is None
    assert result.passed is None
    assert result.suggested == "No face detected in image"


@pytest.mark.parametrize(
    "score, expected_status",
    [
        (_GATE_THRESHOLDS[GateType.FACE][GateStatus.WARNING] + 0.1, GateStatus.PASS),
        ((_GATE_THRESHOLDS[GateType.FACE][GateStatus.FAIL] + _GATE_THRESHOLDS[GateType.FACE][GateStatus.WARNING]) / 2, GateStatus.WARNING),
        (_GATE_THRESHOLDS[GateType.FACE][GateStatus.FAIL] - 0.05, GateStatus.FAIL),
    ],
)
def test_maps_detected_score_to_expected_status(score, expected_status):
    with patch.object(module, "_detect_face_score", return_value=score):
        result = module.face_validator(Image.new("RGB", (2, 2)))

    assert result.score == score
    assert result.passed == (expected_status == GateStatus.PASS)
