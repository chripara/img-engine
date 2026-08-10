# img-engine

> Local, offline image generation engine with pluggable model backends, profile-based recipes, batch generation, prompt refinement, automatic quality gates, and VRAM-safe execution. Built for game asset pipelines and creative production workflows.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is img-engine?

**img-engine** is a local-first image generation engine built on [HuggingFace Diffusers](https://github.com/huggingface/diffusers) with a pluggable backend architecture. Swap diffusion models — SDXL-class, Flux, Stable Diffusion 3.5, or any future checkpoint — without changing the pipeline. It provides a structured interface for generating high-quality images from text prompts.

Image generation itself is 100% local — no cloud inference, no data leaving your machine for the actual diffusion pipeline. Two exceptions to this, both documented in full below: the optional Prompt Refinement Engine ([PRE](#prompt-refinement-engine-pre)), and first-run model downloads for two of the [quality gates](#quality-gates).

Designed initially for generating game card artwork (characters, items, scene backgrounds), it is general-purpose and extensible for any creative or production use case requiring local AI image generation.

---

## Features

- **Local-first image generation** — the diffusion pipeline itself runs 100% locally, no API keys, no cloud inference
- **SDXL-class quality** — runs DreamShaper XL, Juggernaut XL, AlbedoBase XL, and SDXL base
- **Profile-based recipe registry** — each use case (Character, Product, Scene) binds to a complete generation recipe: checkpoint + VAE + scheduler + CFG + steps + native resolution
- **Compel integration** — bypasses the CLIP 77-token limit for long, detailed prompts (~150 tokens)
- **Batch generation** — generate 1–10 images per request (see [Seeds and batches](#seeds-and-batches) for current seed-distinctness caveats)
- **LoRA & style presets** — load style/character LoRA adapters with per-adapter strength, or select a named preset combining LoRA + prompt scaffolding
- **Automatic quality gates** — every generated image is scored for tiling artifacts, prompt adherence, hand/face plausibility, and general visual quality (see [Quality Gates](#quality-gates))
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
│        SDXL Backend             │  ← Diffusers pipeline, Compel, scheduler, VAE, LoRA
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│      Profile Recipe Registry    │  ← ProfileSpec: checkpoint + VAE + scheduler + params
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│         Quality Gates           │  ← tiling, CLIP, hands, face, IQA — run per output image
└─────────────────────────────────┘
```

**Note on the Gradio UI:** the UI builds its own request object from raw form values and sends it as JSON over HTTP to the Flask API — it does not duplicate or bypass validation. The Flask API's `GenerateRequest` schema is the single source of truth; the UI's local schema is intentionally looser since it exists to serialize dropdown/form values, not to enforce the contract.

**Note on the upscaler backends:** `ESRGANBackend` and `LatentDiffusionBackend` share their `load`/`unload` context-manager protocol via a common `BaseBackend.__enter__`/`__exit__` (typed with `Self`), so each backend loads its model exactly once per batch instead of duplicating that logic per subclass. Per-image seed and index are threaded through to the upscaler stage, so upscaled output filenames don't collide even across a seedless batch.

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

**Use a dedicated virtual environment for this project.** Installing into a shared/global Python environment risks dependency conflicts with other projects (observed in practice: a global `torch`/`diffusers`/`mediapipe`/`protobuf` version clash caused by unrelated projects sharing the same site-packages). Always `venv\Scripts\activate` before running or installing anything for img-engine.

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

Response: `{ "images": ["<base64>", ...] }` — each image entry includes a `seed` and a `quality` list of gate results (see [Quality Gates](#quality-gates)).

### Seeds and batches

Current, verified-against-code behavior for `batch_count > 1` (three distinct cases):

- **No `seed` given** → each image is generated with an unseeded (fully random) generator. Output filenames get a positional suffix (`seed_NaN_1.png`, `seed_NaN_2.png`, ...) so files don't collide, and this now applies consistently across the SDXL generation stage **and** the upscaler stage (ESRGAN / generative). However, the actual random value used internally is **not** captured anywhere — the response's `seed` field for these images is `null`, so these specific images cannot be deterministically reproduced later from the response alone.
- **`seed` given, no `spread`** — ⚠️ **known bug, not yet fixed:** every image in the batch currently receives the *exact same* seed value (not `seed`, `seed+1`, `seed+2`, ...). Since the seed is not `null`, the filename-collision workaround above doesn't kick in either — all images in the batch write to the same `seed_<seed>.png` file and overwrite each other. Only the last-generated image survives on disk. Tracked as an open fix.
- **`seed` and `spread` both given** → intentionally *non-deterministic*: each image gets a random value in `[seed - spread, seed + spread]`, freshly randomized on every call via Python's unseeded global `random.randint`. This is a deliberate exploration feature (get variations near a seed) — running the same request twice will **not** produce the same images or the same per-image seed values. Not part of the core SRS contract; a project-specific extension. The upscaler stage now correctly reflects each image's actual resolved seed (not the raw request seed) in its own output filename.

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

## Quality Gates

After each image is generated, it passes through a set of automatic quality checks ("gates"). Each gate returns a score, a pass/fail status, and (if not passing) a suggested reason — surfaced in the API response under `quality` per image.

| Gate | What it checks | How |
|---|---|---|
| `TILING` | Repeating/tiled visual patterns | FFT-based autocorrelation (pure math, no model) |
| `CLIP` | Does the image match the (refined) prompt? | Cosine similarity between CLIP image and text embeddings (`openai/clip-vit-base-patch32`) |
| `HANDS` | Anatomically plausible hands | mediapipe hand landmark detection, combined with geometric plausibility checks (finger segment length ratios, fingertip-to-fingertip fusion distance) |
| `FACE` | Face detected and recognizable | mediapipe face detection confidence |
| `IQA` | General visual quality (noise, blur, compression artifacts) | No-reference deep-learning quality model (`musiq` via `pyiqa`) |

Gates run in parallel per image (`ThreadPoolExecutor`).

### Known limitations (verified, not aspirational)

- **`HANDS` is a heuristic, not a trained classifier.** Raw mediapipe detection confidence alone was found (via manual testing across ~20 generated images) to pass hands with clearly visible finger-count/fusion defects roughly 90% of the time — confidence measures "does this look like a hand-shaped region," not "are the fingers anatomically correct." The current implementation adds geometric plausibility checks on top of detection confidence to catch more of these cases, but the rule thresholds are engineering estimates, not calibrated against labeled data. A trained classifier over labeled hand-quality examples is the natural upgrade path once a golden evaluation set exists (see Roadmap, E08-S05).
- **`HANDS` / `FACE` score of `0` means "nothing detected," not necessarily "detected and deformed."** If a hand or face isn't visible in frame at all (cropped out, occluded, or genuinely absent from the composition — e.g. a close-up shot that doesn't include a face), the gate currently reports the same score/message as a detected-but-malformed case. These two situations aren't yet distinguished in the response.
- **All threshold values (`CLIP`, `HANDS`, `FACE`, `IQA`, and `TILING`) are engineering estimates**, not derived from labeled data. Calibration against a golden evaluation set is planned.
- **First-run network calls.** `CLIP` (via `transformers`) and `IQA` (via `pyiqa`) download pretrained checkpoints from their respective hubs on first use, then cache locally for subsequent runs. This is a one-time exception to the "100% local" claim above — same category as the Groq PRE path.

---

## Development Tools

### Golden-set generator

`utils/generate_golden_set.py` generates a fixed, reproducible sample of images (15 curated prompts × 4 fixed seeds = 60 images) for manually evaluating and calibrating the [quality gates](#quality-gates). Outputs go to a timestamped folder under `output/golden_set/`, alongside a `manifest.json` recording each image's prompt, profile, seed, and the gate scores computed at generation time.

Run it as a module from the project root (not as a direct file path — otherwise Python won't resolve the `app` package):

```powershell
python -m utils.generate_golden_set
```

---

## Known issues

Verified against the current codebase — not aspirational, these are real, open gaps:

- **Batch seed collision** (`seed` given, no `spread`, `num_images > 1`) — see [Seeds and batches](#seeds-and-batches). All images in the batch overwrite the same file. Still open.
- See [Quality Gates → Known limitations](#quality-gates) for gate-specific gaps (IQA threshold scale, HANDS heuristic limits, HANDS/FACE zero-score ambiguity).

---

## Roadmap

- [x] E01 — Local SDXL-class generation
- [x] E02 — Inputs & batching *(long-prompt chunking implemented; remaining stories in progress)*
- [ ] E03 — Outputs (ControlNet, aspect ratios, negatives)
- [ ] E04 — Non-functional (VRAM, reproducibility)
- [ ] E05 — Constraints & interface
- [x] E06 — LoRA & style presets
- [ ] E07 — Contract unification
- [ ] E08 — Quality & acceptance *(quality gates in progress: TILING/CLIP/HANDS/FACE/IQA implemented, threshold calibration and golden-set benchmark — E08-S05 — still pending)*
- [ ] E09 — Output pipeline & quality stages

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference | Stable Diffusion XL (Diffusers) |
| Token handling | Compel |
| Prompt refinement | Groq (Llama 3.3 70B) primary, Mistral 7B via Ollama fallback |
| Quality gates | CLIP (`transformers`), mediapipe (hands/face), `pyiqa` (no-reference IQA) |
| API | Flask |
| UI | Gradio (dev-only) |
| Validation | Pydantic v2 |
| GPU | PyTorch + CUDA |

---

## Licensing

The source code in this repository is MIT licensed (see [`License.md`](./License.md)).

Model weights are **not** covered by this license and are downloaded at runtime from their respective sources. Each carries its own terms — SDXL checkpoints are distributed under CreativeML Open RAIL++-M, which imposes use restrictions. Review the license of any checkpoint, ControlNet, LoRA, or upscaler model before using generated output commercially.

---

## License

MIT — see [`License.md`](./License.md).