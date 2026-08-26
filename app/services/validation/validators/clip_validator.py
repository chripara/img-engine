from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from app.schemas.generate import GateResult
from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, _GATE_MESSAGES
from utils.enums.gate import GateStatus, GateType
import torch

_clip_model = None
_clip_processor = None

def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

def clip_validator(image: Image.Image, prompt: str) -> GateResult:
    model, processor = _load_clip()
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    score = torch.nn.functional.cosine_similarity(
        outputs.image_embeds, outputs.text_embeds
    ).item()

    status = GateStatus.FAIL \
        if score < _GATE_THRESHOLDS[GateType.CLIP][GateStatus.FAIL] \
        else GateStatus.WARNING \
        if score < _GATE_THRESHOLDS[GateType.CLIP][GateStatus.WARNING] \
        else GateStatus.PASS

    return GateResult(
        gate=GateType.CLIP,
        score=score,
        passed=status == GateStatus.PASS,
        suggested=_GATE_MESSAGES[GateType.CLIP][status],
    )