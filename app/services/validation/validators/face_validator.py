from PIL import Image
from app.schemas.generate import GateResult
from utils.enums.gate import GateType, GateStatus
from app.services.validation.registries.validator_registry import _GATE_MESSAGES, resolve_gate_status
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

def _detect_face_score(image: Image.Image) -> float | None:
    detector = _load_face()
    arr = np.array(image.convert("RGB"))
    result = detector.process(arr)

    if not result.detections:
        return None

    return max(d.score[0] for d in result.detections)

def face_validator(image: Image.Image) -> GateResult:
    score = _detect_face_score(image)

    if score is None:
        return GateResult(
            gate = GateType.FACE,
            score = None,
            passed = None,
            suggested = _GATE_MESSAGES[GateType.FACE][GateStatus.NOT_APPLICABLE],
        )

    status = resolve_gate_status(GateType.FACE, score)

    return GateResult(
        gate = GateType.FACE,
        score = score,
        passed = status == GateStatus.PASS,
        suggested = _GATE_MESSAGES[GateType.FACE][status],
    )
