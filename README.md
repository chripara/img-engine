# img-engine

> Local, offline image generation engine with pluggable model backends, profile-based recipes, batch generation, prompt refinement, and VRAM-safe execution. Built for game asset pipelines and creative production workflows.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is img-engine?

**img-engine** is a local-first image generation engine built on [HuggingFace Diffusers](https://github.com/huggingface/diffusers) with a pluggable backend architecture. Swap diffusion models — SDXL-class, Flux, Stable Diffusion 3.5, or any future checkpoint — without changing the pipeline. It provides a structured interface for generating high-quality images from text prompts.

Image generation itself is 100% local — no cloud inference, no data leaving your machine for the actual diffusion pipeline. The optional Prompt Refinement Engine (PRE) is the one exception: see [Prompt Refinement Engine](#prompt-refinement-engine-pre) below for the full, honest picture.

Designed initially for generating game card artwork (characters, items, scene backgrounds), it is general-purpose and extensible for any creative or production use case requiring local AI image generation.

---

## Features

- **Local-first image generation** — the diffusion pipeline itself runs 100% locally, no API keys, no cloud inference
- **SDXL-class quality** — runs DreamShaper XL, Juggernaut XL, AlbedoBase XL, and SDXL base
- **Profile-based recipe registry** — each use case (Character, Product, Scene) binds to a complete generation recipe: checkpoint + VAE + scheduler + CFG + steps + native resolution
- **Compel integration** — bypasses the CLIP 77-token limit for long, detailed prompts (~150 tokens)
- **Batch generation** — generate 1–10 images per request (see [Seeds and batches](#seeds-and-batches) for current seed-distinctness caveats)
- **Prompt Refinement Engine (PRE)** — optional, hybrid: tries Groq-hosted Llama 3.3 70B first, falls back to local Mistral 7B (via Ollama) — see below
- **Pydantic validation** — request schema enforced at the API boundary (prompt max 600 chars, batch N ∈ [1,10])
- **VRAM-safe execution** — context manager lifecycle: load → generate → unload + `torch.cuda.empty_cache()` per batch
- **Seed reproducibility** — explicit `torch.Generator` seeding for deterministic output on a single image; batch-level behavior currently has known gaps (see [Seeds and batches](#seeds-and-batches))
- **Gradio UI** — browser-based interface for local testing and exploration (dev-only, not part of the API contract)
- **Flask REST API** — JSON contract for pipeline integration; the only conformant, validated interface

---

## Architecture

```
┌─────────────────────────────────┐
│         Gradio UI / CLI         │  ← dev-only, out of contract
└────────────────┬────────────────┘
                 │ HTTP POST /generate
┌────────────────▼────────────────┐
│       Flask REST Controller     │  ← Pydantic validation (GenerateRequest) — the actual contract
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│         Image Service           │  ← Batch orchestration, PRE hook
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│         Image Engine            │  ← Context manager (load / generate / unload)
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│        SDXL Backend             │  ← Diffusers pipeline, Compel, scheduler, VAE
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│      Profile Recipe Registry    │  ← ProfileSpec: checkpoint + VAE + scheduler + params
└─────────────────────────────────┘
```

**Note on the Gradio UI:** the UI builds its own request object from raw form values and sends it as JSON over HTTP to the Flask API — it does not duplicate or bypass validation. The Flask API's `GenerateRequest` schema is the single source of truth; the UI's local schema is intentionally looser since it exists to serialize dropdown/form values, not to enforce the contract.

---

## Profiles

| Profile | Use Case | Default Checkpoint | Native Size |
|---|---|---|---|
| `CHARACTER` | Hero / character art | AlbedoBase XL | 1024×1024 |
| `PRODUCT` | Equipment, weapons, relics, icons — isolated objects | DreamShaper XL | 1024×1024 |
| `SCENE_FRAME` | Card frames, backgrounds, environments, logo | Juggernaut XL | 832×1216 (portrait)* |

Each profile carries its own VAE, scheduler, CFG, steps, default negative prompt, and optional refiner — defined as data in the registry, not branching code.

\* *This reflects the current `native_size` value in `profile_registry.py`. Worth double-checking whether portrait is actually intended for a background/environment profile, or whether the registry values are swapped — flagging rather than silently picking one.*

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 5070 Ti, 16 GB VRAM, Blackwell/cu128)
- [Ollama](https://ollama.com/) (optional — local fallback for prompt refinement; app starts fine without it)
- A [Groq](https://groq.com/) API key (optional — only needed if you want the default, higher-quality PRE path; see below)

---

## Installation

```bash
git clone https://github.com/chripara/img-engine.git
cd img-engine
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Download your preferred SDXL checkpoint (`.safetensors`) and set the path via the profile registry.

### Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Optional | Enables the primary (cloud) PRE path. Without it, PRE falls back to local Ollama/Mistral, or passes the prompt through unrefined if neither is available. |
| `OLLAMA_PATH` | Optional | Overrides auto-detection of the Ollama executable. If unset, the app tries `shutil.which("ollama")`, then falls back to assuming `ollama` is on PATH. |

---

## Usage

### Start the server

```bash
python run.py
```

### Gradio UI

Open `http://127.0.0.1:7860` in your browser. Dev/testing convenience only — not part of the API contract.

### REST API

```bash
POST /generate
Content-Type: application/json

{
  "prompt": "tanzanite crystal orb held in an open palm, deep violet aura, fantasy game item",
  "profile": "PRODUCT",
  "feeling": "Mystical & Ethereal",
  "environment": "Dark Dungeon",
  "num_images": 3,
  "refine": false
}
```

Response: `{ "images": ["<base64>", ...] }`

### Seeds and batches

Current, verified-against-code behavior for `batch_count > 1` (three distinct cases):

- **No `seed` given** → each image is generated with an unseeded (fully random) generator. Output filenames get a positional suffix (`seed_NaN_1.png`, `seed_NaN_2.png`, ...) so files don't collide. However, the actual random value used internally is **not** captured anywhere — the response's `seed` field for these images is `null`, so these specific images cannot be deterministically reproduced later from the response alone.
- **`seed` given, no `spread`** — ⚠️ **known bug, not yet fixed:** every image in the batch currently receives the *exact same* seed value (not `seed`, `seed+1`, `seed+2`, ...). Since the seed is not `null`, the filename-collision workaround above doesn't kick in either — all images in the batch write to the same `seed_<seed>.png` file and overwrite each other. Only the last-generated image survives on disk. Tracked as an open fix.
- **`seed` and `spread` both given** → intentionally *non-deterministic*: each image gets a random value in `[seed - spread, spread]`, freshly randomized on every call via Python's unseeded global `random.randint`. This is a deliberate exploration feature (get variations near a seed) — running the same request twice will **not** produce the same images or the same per-image seed values. Not part of the core SRS contract; a project-specific extension.

**Bottom line:** only single-image requests (`num_images = 1`) with an explicit `seed` are currently fully reproducible end-to-end. Batch reproducibility for `seed`-without-`spread` is a known open issue.

---

## Prompt Refinement Engine (PRE)

When `refine: true`, the engine expands short prompts into detailed image descriptions optimized for SDXL, before generation. This path is **hybrid**, not purely local:

1. **First attempt:** [Groq](https://groq.com/)-hosted `llama-3.3-70b-versatile` (cloud API call, requires `GROQ_API_KEY`).
2. **Fallback:** local **Mistral 7B** via Ollama, if Groq fails or `GROQ_API_KEY` isn't set.
3. **Last resort:** the original, unrefined prompt is passed through unchanged if both fail.

This means `refine: true` is **not** a purely local operation by default — it sends your prompt to Groq's cloud API unless you don't have a `GROQ_API_KEY` configured, in which case it's local-only via Ollama. If full-offline operation matters to you, either don't set `GROQ_API_KEY` (Ollama-only fallback) or set `refine: false`.

Ollama itself is managed automatically when available — started on app launch, stopped on exit — but the app no longer fails to start if Ollama isn't installed.

---

## Known issues

Verified against the current codebase — not aspirational, these are real, open gaps:

- **Batch seed collision** (`seed` given, no `spread`, `num_images > 1`) — see [Seeds and batches](#seeds-and-batches). All images in the batch overwrite the same file.
- **Upscaler output filenames can also collide.** The per-image `index` used to disambiguate SDXL-stage filenames is not currently threaded through to the upscaler stage (`ENHANCED`/`GENERATIVE` quality) — upscaled outputs from a seedless batch may overwrite each other the same way.
- **`ESRGANBackend` loads its model twice per upscale call** (once explicitly by the caller, once again inside `upscale()`) — wasted load time, not a correctness bug.

---

## Roadmap

- [x] E01 — Local SDXL-class generation
- [x] E02 — Inputs & batching *(long-prompt chunking implemented; remaining stories in progress)*
- [ ] E03 — Outputs (ControlNet, aspect ratios, negatives)
- [ ] E04 — Non-functional (VRAM, reproducibility)
- [ ] E05 — Constraints & interface
- [x] E06 — LoRA & style presets
- [ ] E07 — Contract unification
- [ ] E08 — Quality & acceptance
- [ ] E09 — Output pipeline & quality stages

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference | Stable Diffusion XL (Diffusers) |
| Token handling | Compel |
| Prompt refinement | Groq (Llama 3.3 70B) primary, Mistral 7B via Ollama fallback |
| API | Flask |
| UI | Gradio (dev-only) |
| Validation | Pydantic v2 |
| GPU | PyTorch + CUDA |

---

## License

MIT — *(license text to be added; this repository does not yet include a `LICENSE` file)*
