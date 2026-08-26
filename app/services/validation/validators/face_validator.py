from PIL import Image
from app.schemas.generate import GateResult
from utils.enums.gate import GateType, GateStatus
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, _GATE_MESSAGES
import numpy as np
import mediapipe as mp

_face_detector = None

def _load_face():
    global _face_detector
    if _face_detector is None:
        _face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5,
        )
    return _face_detector

def face_validator(image: Image.Image) -> GateResult:
    detector = _load_face()
    arr = np.array(image.convert("RGB"))
    result = detector.process(arr)

    if not result.detections:
        return GateResult(
            gate = GateType.FACE,
            score = None,
            passed = None,
            suggested = _GATE_MESSAGES[GateType.FACE][GateStatus.NOT_APPLICABLE],
        )
    else:
        score = max(d.score[0] for d in result.detections)

    status = GateStatus.FAIL \
        if score < _GATE_THRESHOLDS[GateType.FACE][GateStatus.FAIL] \
        else GateStatus.WARNING \
        if score < _GATE_THRESHOLDS[GateType.FACE][GateStatus.WARNING] \
        else GateStatus.PASS

    return GateResult(
        gate=GateType.FACE,
        score=score,
        passed=status == GateStatus.PASS,
        suggested=_GATE_MESSAGES[GateType.FACE][status],
    )

