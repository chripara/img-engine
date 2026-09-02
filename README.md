# img-engine

> Local, offline image generation engine with pluggable model backends, profile-based recipes, batch generation, prompt refinement, automatic quality gates, and VRAM-safe execution. Built for game asset pipelines and creative production workflows.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)
![License](https://img.shields.io/badge/license-MIT-green)

---
## What is img-engine?

**img-engine** is a local-first image generation engine built on [HuggingFace Diffusers](https://github.com/huggingface/diffusers) with a pluggable backend architecture. Swap diffusion models — SDXL-class, Flux, Stable Diffusion 3.5, or any future checkpoint — without changing the pipeline. Designed initially for generating game card artwork (characters, items, scene backgrounds), it is general-purpose and extensible for any creative or production use case requiring local AI image generation.

Image generation itself is 100% local — no cloud inference, no data leaving your machine for the actual diffusion pipeline. Two documented exceptions: the optional [Prompt Refinement Engine](#prompt-refinement-engine-pre), and first-run model downloads for some [quality gates](#quality-gates).

**The part of this project worth your attention isn't "it generates images" — it's the evaluation layer around that.** Every generated image is scored by five automatic quality gates (tiling, prompt-adherence, hand/face plausibility, general visual quality), and there's a dedicated benchmark harness that runs a fixed prompt set across every profile, checkpoint, and LoRA combination to make those gates comparable. That harness and its honest limitations are documented in full below — including where the gates themselves are still uncalibrated, and why that matters.

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

**Note on the Gradio UI:** the UI builds its own request object from raw form values and sends it as JSON over HTTP to the Flask API — it does not duplicate or bypass validation. The Flask API's `GenerateRequest` schema is the single source of truth; the UI's local schema is intentionally independent since it exists to serialize dropdown/form values for local debugging, not to enforce the contract.

**Note on the upscaler backends:** `ESRGANBackend` and `LatentDiffusionBackend` share their `load`/`unload` context-manager protocol via a common `BaseBackend.__enter__`/`__exit__`, so each backend loads its model exactly once per batch. Per-image seed and index are threaded through to the upscaler stage, so upscaled output filenames don't collide even across a seedless batch.

---
## Quickstart

```bash
git clone https://github.com/chripara/img-engine.git
cd img-engine
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python run.py
```

Requires Python 3.10+, a CUDA-capable GPU, and a downloaded SDXL checkpoint referenced from the profile registry. Full requirements, environment variables, the REST API contract, and a known dependency-conflict gotcha are in [`docs/SETUP.md`](./docs/SETUP.md).

---
## Profiles

| Profile | Use Case | Default Checkpoint | Native Size |
|---|---|---|---|
| `CHARACTER` | Hero / character art | AlbedoBase XL | 1024×1024 |
| `PRODUCT` | Equipment, weapons, relics, icons — isolated objects | DreamShaper XL | 1024×1024 |
| `SCENE_FRAME` | Card frames, backgrounds, environments, logo | Juggernaut XL | 832×1216 (portrait)* |

Each profile carries its own checkpoint, VAE, scheduler, CFG, steps, native size, and upscaler — defined as data in the registry, not branching code.

\* *This reflects the current `native_size` value in `profile_registry.py`. Worth double-checking whether portrait is actually intended for a background/environment profile, or whether the registry values are swapped — flagging rather than silently picking one.*

---
## Quality Gates

After each image is generated, it passes through five automatic quality checks. Each gate returns a score, a status (`PASS` / `WARNING` / `FAIL` / `NOT_APPLICABLE`), and (if not passing) a suggested reason — surfaced in the API response under `quality` per image. Gates run in parallel (`ThreadPoolExecutor`).

| Gate | What it checks | How |
|---|---|---|
| `TILING` | Repeating/tiled visual patterns | FFT-based autocorrelation (pure math, no model) |
| `CLIP` | Does the image match the (refined) prompt? | Cosine similarity between CLIP image and text embeddings (`openai/clip-vit-base-patch32`) |
| `HANDS` | Anatomically plausible hands | mediapipe hand landmark detection + pretrained anatomy classifier (`angusleung100/bad-anatomy-realism-classifier`), scores averaged across detected hands |
| `FACE` | Face detected and recognizable | mediapipe face detection confidence |
| `IQA` | General visual quality (noise, blur, compression artifacts) | No-reference deep-learning quality model (`musiq` via `pyiqa`) |

**`NOT_APPLICABLE` status.** `HANDS` and `FACE` return `NOT_APPLICABLE` (`score` and `passed` both `null`) when nothing is detected — no hand or face visible in frame at all. Distinct from a detected-but-malformed result, which is scored normally.

### Development tools built around the gates

**Golden-set generator** (`utils/generate_golden_set.py`) generates a fixed, reproducible sample of images (15 curated prompts × 4 fixed seeds = 60 images) for manually evaluating and calibrating the gates above. Outputs go to a timestamped folder under `output/golden_set/`, alongside a `manifest.json` recording each image's prompt, profile, seed, and gate scores. Sample outputs are committed directly in this repo (not hosted externally) so the results are reviewable without a GPU.

```powershell
python -m utils.generate_golden_set
```

**Cross-profile / LoRA benchmark** (`utils/generate_benchmark.py`) runs a fixed prompt set across every profile, seed, and LoRA on/off variant, scoring each image with the same gates. Outputs go to a timestamped folder under `output_images/benchmark/`, alongside a `manifest.json` recording profile, prompt, seed, LoRA variant, and gate results.

**This validates the benchmarking methodology, not a "best recipe" claim.** Since the gates aren't calibrated yet (see [Known limitations](#known-limitations-verified-not-aspirational)), pass/fail results here aren't proof that one profile or checkpoint is objectively better than another. What it does prove: the infrastructure to make that comparison exists, runs end-to-end, and produces comparable, structured evidence — the prerequisite for a real answer once calibration happens.

```powershell
python -m utils.generate_benchmark
```

Run both as modules from the project root (not as a direct file path — otherwise Python won't resolve the `app` package).

---
## Seeds and batches

Current, verified-against-code behavior for `batch_count > 1` (three distinct cases):

- **No `seed` given** → each image is generated with an unseeded (fully random) generator. Output filenames get a positional suffix (`seed_NaN_1.png`, `seed_NaN_2.png`, ...) so files don't collide. The actual random value used internally is not captured anywhere — the response's `seed` field for these images is `null`, so they cannot be deterministically reproduced from the response alone.
- **`seed` given, no `spread`** → each image in the batch gets a distinct, deterministic seed (`seed`, `seed+1`, `seed+2`, ...). Fully reproducible: the same request produces the same seeds, and each seed maps to its own output file.
- **`seed` and `spread` both given** → intentionally *non-deterministic*: each image gets a random value in `[seed - spread, seed + spread]`, freshly randomized on every call. Deliberate exploration feature (get variations near a seed), not part of the core SRS contract — running the same request twice will not produce the same images.

**Bottom line:** single-image requests and multi-image requests with an explicit `seed` (no `spread`) are both fully reproducible. Only the `spread` case is intentionally non-reproducible, by design.

---
## Prompt Refinement Engine (PRE)

When `refine: true`, the engine expands short prompts into detailed image descriptions optimized for SDXL, before generation. This path is **hybrid**, not purely local:

1. **First attempt:** [Groq](https://groq.com/)-hosted `llama-3.3-70b-versatile` (cloud API call, requires `GROQ_API_KEY`).
2. **Fallback:** local **Mistral 7B** via Ollama, if Groq fails or `GROQ_API_KEY` isn't set.
3. **Last resort:** the original, unrefined prompt is passed through unchanged if both fail.

`refine: true` is **not** purely local by default — it sends your prompt to Groq's cloud API unless `GROQ_API_KEY` is unset, in which case it's local-only via Ollama. Ollama itself is managed automatically when available (started on app launch, stopped on exit) — the app doesn't fail to start if Ollama isn't installed.

---
## Known limitations (verified, not aspirational)

Everything below was checked directly against the current code and current benchmark data — not a hedge, an actual account of what this system can and can't prove about itself yet.

- **The quality gates are not yet calibrated.** All threshold values (`CLIP`, `HANDS`, `FACE`, `IQA`, `TILING`) are engineering estimates, not derived from labeled data. This means the benchmark below can show *consistency* across profiles/checkpoints/LoRA, but not yet *proof* that one recipe is objectively better than another — that requires calibrating the gates against a labeled golden set first.
- **`HANDS` classifier reliability.** The anatomy classifier (`angusleung100`) was fine-tuned on a small dataset (~134 images) and, across manual testing on dozens of generations, doesn't meaningfully discriminate hand quality for this project's art style and pose distribution — grip/weapon-holding poses in particular. Treat the `HANDS` score as experimental, not a trustworthy signal.
- **`FACE` detector domain mismatch.** mediapipe's face detector (BlazeFace) is trained exclusively on real photographs, per its official model card — not illustrated or stylized art. Observed false positives on symmetric, paired decorative hardware (a sword's crossguard/pommel; a book's brass clasp) suggest it can misfire on this project's fantasy-illustration style. Treat `FACE` results as directional, not ground truth.
- **`CLIP` threshold is lenient relative to observed scores.** Across benchmark data, CLIP scores cluster around 0.29–0.39, comfortably above the current 0.20 fail / 0.25 warning thresholds — in practice this gate has not yet failed a real generation, which limits how much it's currently telling us.
- **Batch seed collision (`seed` given, no `spread`) is fixed but wasn't always.** Each image in a batch now gets a distinct, deterministic seed (`seed`, `seed+1`, `seed+2`, ...) — see [Seeds and batches](#seeds-and-batches) for the full breakdown of all three seed/spread cases.
- **First-run network calls.** `CLIP`/`HANDS` (via `transformers`) and `IQA` (via `pyiqa`) download pretrained checkpoints from their respective hubs on first use, then cache locally. One-time exception to the "100% local" claim above — same category as the Groq PRE path.

---
## Licensing

The source code in this repository is MIT licensed — see [`LICENSE`](./LICENSE).

Model weights are **not** covered by this license and are downloaded at runtime from their respective sources. Each carries its own terms — SDXL checkpoints are distributed under CreativeML Open RAIL++-M, which imposes use restrictions. Review the license of any checkpoint, ControlNet, LoRA, or upscaler model before using generated output commercially.
