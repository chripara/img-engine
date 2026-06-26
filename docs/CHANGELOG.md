# Changelog

All notable changes to img-engine are documented here.

---

## [Unreleased]

---

## E02 — Inputs & Batching

### US-IMG-E02-S04: Seed reproducibility
- Explicit `torch.Generator` seeding per request
- `seed: int | None` field in `GenerateRequest` (range 0–2³²-1)
- `spread: int` field (default 0) — controls seed variation across batch: each image gets `seed ± random(0, spread)`
- `spread=0` → all images identical (full reproducibility); `spread>0` → exploration around base seed
- UI: Use Seed checkbox + Seed number input + Spread number input
- Same seed + same spread on the same device yields identical batch

### US-IMG-E02-S03: Batch per description
- `num_images: int` field in `GenerateRequest` (range 1–10, validated via Pydantic)
- Image service orchestrates N calls to the engine
- Response returns `list[bytes]` encoded as base64 JSON array
- Gradio Gallery displays batch output with thumbnails

### US-IMG-E02-S02: Free-form text prompts
- `prompt` field enforced with `max_length=600` via Pydantic `Field`
- Over-limit input rejected with 422
- 600 chars maps to ~150 CLIP tokens (Compel multi-chunk limit)

### US-IMG-E02-S01: Prompt refinement hook
- `refine: bool` field in `GenerateRequest`
- When `True`, prompt is sent to Mistral 7B (via Ollama) before inference
- Ollama lifecycle managed automatically on app start/exit

---

## E01 — Local SDXL-class Generation

### US-IMG-E01-S04: Model/Profile recipe registry
- `ProfileSpec` dataclass: checkpoint, VAE, scheduler, steps, CFG, native size, default negative
- `_PROFILES` dict as single source of truth for all profiles
- Profiles: `CHARACTER`, `PRODUCT`, `SCENE_FRAME`
- VAE and scheduler loaded explicitly per profile
- Local `.safetensors` checkpoint loading supported

### US-IMG-E01-S03: 100% local / offline
- `HF_HUB_OFFLINE` honored
- No network calls during inference
- All weights pre-fetched to local paths

### US-IMG-E01-S02: SDXL checkpoint selectable
- Multiple SDXL-class checkpoints supported: DreamShaper XL, Juggernaut XL, AlbedoBase XL, SDXL base
- Checkpoint selection via profile registry, no source changes required

### US-IMG-E01-S01: User prompts to images
- Text prompt → PNG inference via `StableDiffusionXLPipeline`
- Compel integration for CLIP 77-token limit bypass (~150 tokens)
- `ImageEngine` context manager: load → generate → unload + VRAM cleanup per batch
- Output returned as `bytes` (PNG)
- Gradio UI for local testing
- Flask REST API (`POST /generate`) as canonical interface
