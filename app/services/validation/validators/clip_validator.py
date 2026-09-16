from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from app.schemas.generate import GateResult
from app.services.validation.registries.validator_registry import _GATE_MESSAGES, resolve_gate_status
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

def _compute_clip_score(image: Image.Image, prompt: str) -> float:
    model, processor = _load_clip()
    tokenizer = processor.tokenizer

    max_len = tokenizer.model_max_length - 2  # room for BOS/EOS special tokens
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    chunks = [token_ids[i:i + max_len] for i in range(0, len(token_ids), max_len)] or [[]]

    image_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        chunk_embeds = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk) if chunk else ""
            text_inputs = processor(
                text=[chunk_text], return_tensors="pt",
                padding=True, truncation=True, max_length=tokenizer.model_max_length,
            )
            text_embeds = model.get_text_features(**text_inputs)
            chunk_embeds.append(text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True))

        avg_text_embeds = torch.stack(chunk_embeds).mean(dim=0)

    return torch.nn.functional.cosine_similarity(image_embeds, avg_text_embeds).item()

def clip_validator(image: Image.Image, prompt: str) -> GateResult:
    score = _compute_clip_score(image, prompt)
    status = resolve_gate_status(GateType.CLIP, score)

    return GateResult(
        gate=GateType.CLIP,
        score=score,
        passed=status == GateStatus.PASS,
        suggested=_GATE_MESSAGES[GateType.CLIP][status],
    )
