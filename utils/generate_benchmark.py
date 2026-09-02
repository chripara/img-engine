"""
Benchmark generator for US-IMG-E08-S05 (revised scope).

3 prompts x 3 profiles x 2 seeds x 2 LoRA variants (with FANTASY @ 0.9
vs without) = 36 images. Every prompt runs under every profile, using
that profile's own already-configured recipe -- nothing overridden,
nothing mutated -- once with the FANTASY style LoRA applied and once
without, to compare its effect on gate scores with actual evidence.
"""

import base64
import json
from datetime import datetime
from pathlib import Path

from app.schemas.generate import GenerateRequest
from app.services.pipeline_service import PipelineService
from utils.enums.profile import Profile
from utils.enums.style_presets import StylePreset

# 3 prompts, each originally written for one profile's use case
# (native_profile), but run under all 3 profiles below.
PROMPTS = [
    {"native_profile": Profile.CHARACTER, "text": "elven ranger holding a longbow, both hands gripping the string, detailed face"},
    {"native_profile": Profile.PRODUCT, "text": "ornate golden crown on a velvet cushion, studio lighting, isolated background"},
    {"native_profile": Profile.SCENE_FRAME, "text": "dense mystical forest background, dappled sunlight through leaves, no characters"},
]

PROFILES = [Profile.CHARACTER, Profile.PRODUCT, Profile.SCENE_FRAME]

SEEDS = [1000, 2000]

LORA_VARIANTS = [
    {"label": "with_lora", "style_preset": StylePreset.FANTASY, "lora_strength": 0.9},
    {"label": "no_lora", "style_preset": None, "lora_strength": None},
]


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    out_dir = PROJECT_ROOT / "output_images" / "benchmark" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = len(PROMPTS) * len(PROFILES) * len(SEEDS) * len(LORA_VARIANTS)
    count = 0

    for prompt_def in PROMPTS:
        for profile in PROFILES:
            for seed in SEEDS:
                for variant in LORA_VARIANTS:
                    req = GenerateRequest(
                        profile=profile,
                        num_images=1,
                        prompt=prompt_def["text"],
                        negative_prompt=None,
                        subject=None,
                        environment=None,
                        feeling=None,
                        refine=False,
                        seed=seed,
                        spread=None,
                        style_preset=variant["style_preset"],
                        lora_strength=variant["lora_strength"],
                    )
                    result = PipelineService.generation_pipeline(req)
                    image_result = result.images[0]

                    filename = f"{profile.value}_native-{prompt_def['native_profile'].value}_seed{seed}_{variant['label']}.png"
                    image_bytes = base64.b64decode(image_result.image)
                    (out_dir / filename).write_bytes(image_bytes)

                    manifest.append({
                        "profile": profile.value,
                        "prompt_native_profile": prompt_def["native_profile"].value,
                        "is_native_pairing": profile == prompt_def["native_profile"],
                        "prompt": prompt_def["text"],
                        "seed": seed,
                        "lora_variant": variant["label"],
                        "filename": filename,
                        "gates": [g.model_dump() for g in image_result.quality] if image_result.quality else [],
                    })
                    count += 1
                    print(f"[{count}/{total}] native-{prompt_def['native_profile'].value} / {profile.value} / seed{seed} / {variant['label']} -> {filename}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {count} images saved to {out_dir}")


if __name__ == "__main__":
    main()