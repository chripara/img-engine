from PIL import Image
from app.schemas.generate import GateResult
from utils.enums.gate import GateStatus, GateType
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, _GATE_MESSAGES
import numpy as np
import mediapipe as mp

_hands_detector = None

def _load_hands():
    global _hands_detector
    if _hands_detector is None:
        _hands_detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
        )
    return _hands_detector

def hands_validator(image: Image.Image) -> GateResult:
    detector = _load_hands()
    arr = np.array(image.convert("RGB"))
    result = detector.process(arr)

    if not result.multi_hand_landmarks:
        score = 0.0
    else:
        confidences = [h.classification[0].score for h in result.multi_handedness]
        score = sum(confidences) / len(confidences)

    status = GateStatus.FAIL \
        if score < _GATE_THRESHOLDS[GateType.HANDS][GateStatus.FAIL] \
        else GateStatus.WARNING \
        if score < _GATE_THRESHOLDS[GateType.HANDS][GateStatus.WARNING] \
        else GateStatus.PASS

    return GateResult(
        gate=GateType.HANDS,
        score=score,
        passed=status == GateStatus.PASS,
        suggested=_GATE_MESSAGES[GateType.HANDS][status],
    )