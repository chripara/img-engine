from PIL import Image
import numpy as np
import mediapipe as mp
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from app.schemas.generate import GateResult
from utils.enums.gate import GateStatus, GateType
from app.services.validation.registries.validator_registry import _GATE_MESSAGES, resolve_gate_status

_MAX_NUM_HANDS = 4
_HAND_CROP_PADDING_RATIO = 0.5
_GOOD_ANATOMY_LABELS = {"Realistic_Good_Anatomy", "Unrealistic_Good_Anatomy"}

_anatomy_classifier = None

def _load_anatomy_classifier():
    global _anatomy_classifier
    if _anatomy_classifier is None:
        model = AutoModelForImageClassification.from_pretrained(
            "angusleung100/bad-anatomy-realism-classifier"
        )
        image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        _anatomy_classifier = pipeline(
            "image-classification",
            model=model,
            image_processor=image_processor,
        )
    return _anatomy_classifier


class HandsValidator:
    def __init__(self):
        self._detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=_MAX_NUM_HANDS,
            min_detection_confidence=0.5,
        )
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            self._classifier = _load_anatomy_classifier()
        return self._classifier

    def _detect_hands(self, image: Image.Image) -> list[Image.Image]:
        arr = np.array(image.convert("RGB"))
        result = self._detector.process(arr)

        if not result.multi_hand_landmarks:
            return []

        width, height = image.size
        crops = []

        for hand_landmarks in result.multi_hand_landmarks:
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            pad_x = (x_max - x_min) * _HAND_CROP_PADDING_RATIO
            pad_y = (y_max - y_min) * _HAND_CROP_PADDING_RATIO

            left   = max(0, int((x_min - pad_x) * width))
            right  = min(width, int((x_max + pad_x) * width))
            top    = max(0, int((y_min - pad_y) * height))
            bottom = min(height, int((y_max + pad_y) * height))

            crops.append(image.crop((left, top, right, bottom)))

        return crops

    def _classify_hand(self, crop: Image.Image) -> tuple[float, str, float]:
        predictions = self._get_classifier()(crop)
        scores_by_label = {p["label"]: p["score"] for p in predictions}

        good_score = sum(scores_by_label.get(label, 0.0) for label in _GOOD_ANATOMY_LABELS)
        top = max(predictions, key=lambda p: p["score"])

        return good_score, top["label"], top["score"]

    def _raw_hand_scores(self, image: Image.Image) -> list[tuple[float, str, float]]:
        crops = self._detect_hands(image)
        return [self._classify_hand(c) for c in crops]

    def validate(self, image: Image.Image) -> GateResult:
        per_hand = self._raw_hand_scores(image)

        if not per_hand:
            return GateResult(
                gate=GateType.HANDS,
                score=None,
                passed=None,
                suggested=_GATE_MESSAGES[GateType.HANDS][GateStatus.NOT_APPLICABLE],
            )

        score = sum(r[0] for r in per_hand) / len(per_hand)
        worst = min(per_hand, key=lambda r: r[0])

        status = resolve_gate_status(GateType.HANDS, score)
        base_message = _GATE_MESSAGES[GateType.HANDS][status]
        suggested = f"{base_message} (model: '{worst[1]}', p={worst[2]:.2f})" if base_message else None

        return GateResult(
            gate=GateType.HANDS,
            score=score,
            passed=status == GateStatus.PASS,
            suggested=suggested,
        )

_validator: HandsValidator | None = None

def _get_validator() -> HandsValidator:
    global _validator
    if _validator is None:
        _validator = HandsValidator()
    return _validator

def hands_validator(image: Image.Image) -> GateResult:
    return _get_validator().validate(image)
