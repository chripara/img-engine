from unittest.mock import patch
import pytest
from PIL import Image

from app.services.validation import validator as module
from app.schemas.generate import GateResult
from utils.enums.gate import GateType


def _fake_result(gate: GateType) -> GateResult:
    return GateResult(gate=gate, score=1.0, passed=True, suggested=None)


def test_validate_calls_all_five_validators_and_returns_their_results_in_order():
    image = Image.new("RGB", (2, 2))
    prompt = "a cat"

    with patch.object(module, "tiling_validator", return_value=_fake_result(GateType.TILING)) as m_tiling, \
         patch.object(module, "clip_validator", return_value=_fake_result(GateType.CLIP)) as m_clip, \
         patch.object(module, "hands_validator", return_value=_fake_result(GateType.HANDS)) as m_hands, \
         patch.object(module, "face_validator", return_value=_fake_result(GateType.FACE)) as m_face, \
         patch.object(module, "iqa_validator", return_value=_fake_result(GateType.IQA)) as m_iqa:

        results = module.validate(image, prompt)

    m_tiling.assert_called_once_with(image)
    m_clip.assert_called_once_with(image, prompt)
    m_hands.assert_called_once_with(image)
    m_face.assert_called_once_with(image)
    m_iqa.assert_called_once_with(image)

    assert [r.gate for r in results] == [
        GateType.TILING, GateType.CLIP, GateType.HANDS, GateType.FACE, GateType.IQA,
    ]


def test_validate_propagates_exception_from_any_single_validator():
    """Τεκμηριωμένο ΤΡΕΧΟΝ design — όχι διορθωμένο εδώ. Αν ΕΝΑΣ validator
    σκάσει, ολόκληρο το validate() σκάει (κανένα graceful degradation)."""
    with patch.object(module, "tiling_validator", return_value=_fake_result(GateType.TILING)), \
         patch.object(module, "clip_validator", side_effect=RuntimeError("model failed")), \
         patch.object(module, "hands_validator", return_value=_fake_result(GateType.HANDS)), \
         patch.object(module, "face_validator", return_value=_fake_result(GateType.FACE)), \
         patch.object(module, "iqa_validator", return_value=_fake_result(GateType.IQA)):

        with pytest.raises(RuntimeError, match="model failed"):
            module.validate(Image.new("RGB", (2, 2)), "a cat")
