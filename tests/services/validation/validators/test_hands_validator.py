from unittest.mock import patch
import pytest
from PIL import Image

from app.services.validation.validators.hands_validator import HandsValidator, hands_validator
from app.services.validation.validators import hands_validator as module
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS
from utils.enums.gate import GateType, GateStatus


def _bare_validator() -> HandsValidator:
    return HandsValidator.__new__(HandsValidator)


def test_returns_not_applicable_when_no_hands_detected():
    validator = _bare_validator()
    with patch.object(validator, "_raw_hand_scores", return_value=[]):
        result = validator.validate(Image.new("RGB", (2, 2)))

    assert result.score is None
    assert result.passed is None
    assert result.suggested == "No hands detected in image"


def test_score_is_average_across_detected_hands():
    validator = _bare_validator()
    per_hand = [(0.9, "Realistic_Good_Anatomy", 0.9), (0.5, "Some_Bad_Label", 0.5)]
    with patch.object(validator, "_raw_hand_scores", return_value=per_hand):
        result = validator.validate(Image.new("RGB", (2, 2)))

    assert result.score == pytest.approx(0.7)


def test_suggested_message_names_the_worst_scoring_hand_when_not_passing():
    validator = _bare_validator()
    fail_score = _GATE_THRESHOLDS[GateType.HANDS][GateStatus.FAIL] - 0.1
    per_hand = [(fail_score, "Bad_Anatomy_Label", 0.77)]
    with patch.object(validator, "_raw_hand_scores", return_value=per_hand):
        result = validator.validate(Image.new("RGB", (2, 2)))

    assert "Bad_Anatomy_Label" in result.suggested
    assert "0.77" in result.suggested


@pytest.mark.parametrize(
    "score, expected_status",
    [
        (_GATE_THRESHOLDS[GateType.HANDS][GateStatus.WARNING] + 0.1, GateStatus.PASS),
        ((_GATE_THRESHOLDS[GateType.HANDS][GateStatus.FAIL] + _GATE_THRESHOLDS[GateType.HANDS][GateStatus.WARNING]) / 2, GateStatus.WARNING),
        (_GATE_THRESHOLDS[GateType.HANDS][GateStatus.FAIL] - 0.05, GateStatus.FAIL),
    ],
)
def test_maps_average_score_to_expected_status(score, expected_status):
    validator = _bare_validator()
    with patch.object(validator, "_raw_hand_scores", return_value=[(score, "label", score)]):
        result = validator.validate(Image.new("RGB", (2, 2)))

    assert result.passed == (expected_status == GateStatus.PASS)


def test_module_level_wrapper_uses_lazy_singleton_and_delegates_to_validate():
    with patch.object(module, "_validator", None), \
         patch.object(module, "HandsValidator") as fake_cls:
        fake_instance = fake_cls.return_value
        fake_instance.validate.return_value = "the-result"

        result1 = hands_validator(Image.new("RGB", (2, 2)))
        result2 = hands_validator(Image.new("RGB", (2, 2)))

    fake_cls.assert_called_once()  # singleton — μόνο 1 φορά constructed
    assert result1 == "the-result" and result2 == "the-result"
