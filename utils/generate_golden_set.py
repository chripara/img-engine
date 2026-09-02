"""
Generates a fixed golden-set sample for quality-gate evaluation.

Runs a curated list of 15 prompts (covering CHARACTER/PRODUCT/SCENE_FRAME
profiles) across 4 fixed seeds each (60 images total), saves the images
plus a manifest.json with per-image metadata and the automatically
computed gate scores. Human labels are added separately afterwards
(labels.json), for threshold calibration (E08-S05).
"""

import base64
import json
from datetime import datetime
from pathlib import Path

from app.schemas.generate import GenerateRequest
from app.services.pipeline_service import PipelineService
from utils.enums.profile import Profile

# Curated prompt set: 7 CHARACTER (hands/faces), 4 PRODUCT (no hands/face),
# 4 SCENE_FRAME (textures/gradients, tiling-prone).
PROMPTS = [
    # CHARACTER — hands/faces (7)
    {"id": 1, "profile": Profile.CHARACTER, "text": "elven ranger holding a longbow, both hands gripping the string, detailed face"},
    {"id": 2, "profile": Profile.CHARACTER, "text": "photorealistic chef chopping vegetables, close-up on hands and knife"},
    {"id": 3, "profile": Profile.CHARACTER, "text": "knight in full plate armor raising a sword with both hands, helmet visor up showing face"},
    {"id": 4, "profile": Profile.CHARACTER, "text": "blacksmith hammering a blade on an anvil, muscular forearms and hands gripping hammer"},
    {"id": 5, "profile": Profile.CHARACTER, "text": "wizard casting a spell, hands raised with glowing magic energy between fingers"},
    {"id": 6, "profile": Profile.CHARACTER, "text": "portrait of an old sailor, weathered face, hands resting on a ship's wheel"},
    {"id": 7, "profile": Profile.CHARACTER, "text": "assassin holding twin daggers crossed in front of chest, face partially covered by hood"},

    # PRODUCT — isolated objects, no hands/face (4)
    {"id": 8, "profile": Profile.PRODUCT, "text": "ornate golden crown on a velvet cushion, studio lighting, isolated background"},
    {"id": 9, "profile": Profile.PRODUCT, "text": "enchanted health potion bottle glowing red, sitting on a wooden table"},
    {"id": 10, "profile": Profile.PRODUCT, "text": "steel battle axe, isolated on plain background, product photography style"},
    {"id": 11, "profile": Profile.PRODUCT, "text": "ancient leather-bound spellbook closed with a metal clasp, isolated"},

    # SCENE_FRAME — textures/gradients, tiling-prone (4)
    {"id": 12, "profile": Profile.SCENE_FRAME, "text": "dense mystical forest background, dappled sunlight through leaves, no characters"},
    {"id": 13, "profile": Profile.SCENE_FRAME, "text": "stone dungeon wall texture, moss and cracks, torch-lit"},
    {"id": 14, "profile": Profile.SCENE_FRAME, "text": "abstract magical energy gradient background, purple and blue swirls"},
    {"id": 15, "profile": Profile.SCENE_FRAME, "text": "desert dune landscape at sunset, wide gradient sky"},
]

# Fixed seeds reused for every prompt, so the whole set is fully reproducible
# (same prompt + same seed -> same image, every time this script runs).
SEEDS = [1000, 2000, 3000, 4000]


def main():
    # Each run gets its own timestamped output folder, so old runs are
    # never overwritten and can be compared side by side later.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # utils/generate_golden_set.py → root
    
    out_dir = PROJECT_ROOT / "output" / "golden_set" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    total = len(PROMPTS) * len(SEEDS)
    count = 0

    for prompt_def in PROMPTS:
        for seed in SEEDS:
            # num_images=1, spread=None: one deterministic image per
            # (prompt, seed) pair — no batch randomness involved here.
            req = GenerateRequest(
                profile=prompt_def["profile"],
                num_images=1,
                prompt=prompt_def["text"],
                negative_prompt=None,
                subject=None,
                environment=None,
                feeling=None,
                refine=False,
                seed=seed,
                spread=None,
            )
            result = PipelineService.generation_pipeline(req)
            image_result = result.images[0]

            # Filename encodes prompt_id + seed directly, so the file is
            # self-describing without needing to open manifest.json.
            filename = f"p{prompt_def['id']:02d}_seed{seed}.png"
            image_bytes = base64.b64decode(image_result.image)
            (out_dir / filename).write_bytes(image_bytes)

            # Record everything needed for later human labeling and
            # calibration: prompt, profile, seed, filename, and the
            # gate scores the system already computed during generation.
            manifest.append({
                "prompt_id": prompt_def["id"],
                "prompt": prompt_def["text"],
                "profile": prompt_def["profile"].value,
                "seed": seed,
                "filename": filename,
                "gates": [g.model_dump() for g in image_result.quality] if image_result.quality else [],
            })
            count += 1
            print(f"[{count}/{total}] saved {filename}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {count} images saved to {out_dir}")


if __name__ == "__main__":
    main()

