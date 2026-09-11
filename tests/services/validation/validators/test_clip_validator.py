from unittest.mock import patch
import pytest
from PIL import Image

from app.services.validation.validators import clip_validator as module
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS
from utils.enums.gate import GateType, GateStatus


@pytest.mark.parametrize(
    "score, expected_status",
    [
        (_GATE_THRESHOLDS[GateType.CLIP][GateStatus.WARNING] + 0.1, GateStatus.PASS),
        ((_GATE_THRESHOLDS[GateType.CLIP][GateStatus.FAIL] + _GATE_THRESHOLDS[GateType.CLIP][GateStatus.WARNING]) / 2, GateStatus.WARNING),
        (_GATE_THRESHOLDS[GateType.CLIP][GateStatus.FAIL] - 0.05, GateStatus.FAIL),
    ],
)
def test_clip_validator_maps_score_to_expected_status(score, expected_status):
    with patch.object(module, "_compute_clip_score", return_value=score):
        result = module.clip_validator(Image.new("RGB", (2, 2)), "a cat")

    assert result.gate == GateType.CLIP
    assert result.score == score
    assert result.passed == (expected_status == GateStatus.PASS)
    if expected_status != GateStatus.PASS:
        assert result.suggested is not None


def test_clip_validator_passes_image_and_prompt_to_raw_score_function():
    with patch.object(module, "_compute_clip_score", return_value=0.9) as fake_score:
        image = Image.new("RGB", (2, 2))
        module.clip_validator(image, "a cat in a hat")
    fake_score.assert_called_once_with(image, "a cat in a hat")
