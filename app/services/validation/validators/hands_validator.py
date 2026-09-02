from PIL import Image
import numpy as np
import uuid
import os
import mediapipe as mp
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from app.schemas.generate import GateResult
from utils.enums.gate import GateStatus, GateType
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, _GATE_MESSAGES
from pathlib import Path

_MAX_NUM_HANDS = 4  # documented limit — see README: hand validation covers up to 4 hands/image
_HAND_CROP_PADDING_RATIO = 0.5
_GOOD_ANATOMY_LABELS = {"Realistic_Good_Anatomy", "Unrealistic_Good_Anatomy"}

_model = AutoModelForImageClassification.from_pretrained(
    "angusleung100/bad-anatomy-realism-classifier"
)
_image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

class HandsValidator:
    def __init__(self):
        self._detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=_MAX_NUM_HANDS,
            min_detection_confidence=0.5,
        )
        self._classifier = pipeline(
            "image-classification",
            model=_model,
            image_processor=_image_processor,
        )

    def _detect_hands(self, image: Image.Image) -> list[Image.Image]:

        """Runs mediapipe hand detection and returns padded crops, one per detected hand."""

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

        """Runs the anatomy classifier on one hand crop.
        Returns (marginalized_good_anatomy_score, top_predicted_label, top_predicted_prob)."""

        predictions = self._classifier(crop)
        scores_by_label = {p["label"]: p["score"] for p in predictions}

        good_score = sum(scores_by_label.get(label, 0.0) for label in _GOOD_ANATOMY_LABELS)
        top = max(predictions, key=lambda p: p["score"])

        return good_score, top["label"], top["score"]

    def validate(self, image: Image.Image) -> GateResult:
        crops = self._detect_hands(image)

        if not crops:
            return GateResult(
                gate=GateType.HANDS,
                score=None,
                passed=None,
                suggested=_GATE_MESSAGES[GateType.HANDS][GateStatus.NOT_APPLICABLE],
            )

        per_hand = [self._classify_hand(c) for c in crops]
        score = sum(r[0] for r in per_hand) / len(per_hand)
        worst = min(per_hand, key=lambda r: r[0])

        status = GateStatus.FAIL \
            if score < _GATE_THRESHOLDS[GateType.HANDS][GateStatus.FAIL] \
            else GateStatus.WARNING \
            if score < _GATE_THRESHOLDS[GateType.HANDS][GateStatus.WARNING] \
            else GateStatus.PASS

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

    """Thin module-level wrapper — keeps validator.py's existing
    `executor.submit(hands_validator, image)` call site unchanged."""

    return _get_validator().validate(image)