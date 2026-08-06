# img-engine

**A local SDXL generation engine for game asset pipelines.** Each use case binds to a
complete, pinned recipe — checkpoint, VAE, scheduler, CFG, steps — instead of one model
with hand-tuned parameters per call. Built to produce card artwork at consistent quality
without re-deriving settings every time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-ee4c2c?logo=pytorch)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)
![License](https://img.shields.io/badge/license-MIT-green)

<!-- TODO: 30-second GIF of the Gradio UI generating a batch. This is the highest-value
     thing you can add to this README — most readers decide in the first 15 seconds. -->

---

## The problem

Diffusion output quality depends less on the prompt than on the *combination* of
checkpoint, VAE, scheduler, CFG, and step count. Those combinations aren't
interchangeable: settings that produce clean character art produce mush on an isolated
object. In practice you end up with a spreadsheet of settings and a lot of re-derivation.

img-engine turns that spreadsheet into a registry. You pick a **profile** — the intent
— and the engine applies the full recipe that intent was tuned for.

---

## What it does

| | |
|---|---|
| **Profile-based recipes** | Three profiles, each bound to its own checkpoint, VAE, scheduler, CFG, and step count — as data, not branching code |
| **Pluggable backends** | `BaseBackend` ABC. Adding a Flux or SD3.5 backend requires no pipeline changes |
| **ControlNet guidance** | Canny, depth, pose, scribble — up to 3 simultaneous, with per-control strength |
| **LoRA style presets** | 8 named presets, fused at load time |
| **Batch generation** | 1–10 images per request, with seed control (see [Seeds and batches](#seeds-and-batches)) |
| **Long prompts** | Compel chunking bypasses the CLIP 77-token limit (~150 tokens usable) |
| **Upscaling** | ESRGAN (`enhanced`) or latent diffusion x4 (`generative`) |
| **Quality gates** | Tiling detection via FFT autocorrelation — 1 of 5 planned gates, see [Limitations](#limitations) |
| **Prompt refinement** | Optional. Groq Llama 3.3 70B → local Mistral 7B → passthrough |
| **VRAM lifecycle** | Context managers: load → generate → unload → `empty_cache()` on every stage |

---

## Architecture

```
  Gradio UI                                   dev-only, out of contract
      │  HTTP POST /generate
      ▼
  Flask + Pydantic  ─────────────────────►    the contract surface
      │                                       GenerateRequest is the single
      ▼                                       source of truth for validation
  PipelineService   ─────────────────────►    orchestration
      │                                       ├─ prompt refinement  ┐ parallel
      │                                       └─ ControlNet preproc ┘
      ▼
  ┌───────────────┬──────────────┬───────────────┐
  │ Image         │ Guidance     │ Upscaler      │   each: service → engine → backend
  │ SDXLBackend   │ ControlNet   │ ESRGAN/Latent │   each: context-managed lifecycle
  └───────────────┴──────────────┴───────────────┘
      │
      ▼
  Validator ────────────────────────────────►   quality gates, advisory
      │
      ▼
  ProfileRegistry ──────────────────────────►   checkpoint + VAE + scheduler + CFG + steps
```

Every subsystem follows the same shape: **service → engine → backend → registry**. The
service orchestrates, the engine owns the resource lifecycle, the backend does the work,
the registry holds configuration as data.

---

## Profiles

| Profile | Use case | Checkpoint | Scheduler | Steps | CFG |
|---|---|---|---|---|---|
| `CHARACTER` | Hero and character art | AlbedoBase XL | Euler Discrete | 30 | 7.0 |
| `PRODUCT` | Weapons, relics, icons — isolated objects | DreamShaper XL | Euler Discrete | 30 | 7.0 |
| `SCENE_FRAME` | Frames, backgrounds, environments | Juggernaut XL | DPM++ Multistep | 35 | 4.5 |

`PRODUCT` additionally uses the `madebyollin/sdxl-vae-fp16-fix` VAE and the anime ESRGAN
variant for upscaling; the others use their checkpoint's bundled VAE.

**Resolutions** are request-driven, not profile-driven: `square` 1024×1024 ·
`landscape` 1344×768 · `portrait` 768×1344 · `card_portrait` 832×1216 · `card_large` 1152×896.

---

## Design decisions

The non-obvious choices and what they cost.

**Profile is the unit of selection — checkpoint is not exposed.**
The API deliberately has no `checkpoint` field. Letting a caller pick Juggernaut while
keeping AlbedoBase's scheduler and CFG would defeat the point: each model runs in the
recipe it was tuned for, or not at all. Trade-off: using a checkpoint outside its
profile requires a registry edit.

**Full load/unload per request, no instance cache.**
Optimizes for VRAM headroom over latency. A 16 GB card can then run SDXL + up to 3
ControlNets + an upscaler without swapping. Cost: ~20–40 s of model loading on every
request. An instance cache is the single largest available speedup and is planned.

**LoRA is fused, not kept as a separate adapter.**
`fuse_lora()` bakes the weights into the UNet — faster at inference, and irreversible.
That's correct *because* of the decision above: the pipeline is destroyed after each
request, so there is nothing to unfuse. **These two decisions are coupled** — adding an
instance cache would require either `unfuse_lora()` or a cache key that includes the
preset, meaning N pipelines resident in VRAM.

**`spread` instead of `seed + i` for batches.**
A batch is an exploration tool, not a reproduction tool. `spread` samples seeds from a
window around a base seed, giving related-but-varied outputs. This is deliberately
non-deterministic across calls — see [Seeds and batches](#seeds-and-batches).

**The HTTP API is the contract; there is no CLI.**
One conformant surface rather than two to keep in sync. The Gradio UI is a dev client
with no guarantees, and its request schema is intentionally looser — it serializes
dropdown values, it does not enforce the contract. Evaluation and tests call
`PipelineService` directly rather than going over HTTP.

**Invalid `aspect_ratio` coerces to square rather than rejecting.**
Graceful degradation for a field where a sensible default exists. Note this is
inconsistent with the over-length prompt path, which rejects — an explicit `warnings[]`
field would reconcile the two and is planned.

---

## API

### `POST /generate`

```json
{
  "prompt": "tanzanite crystal orb held in an open palm, deep violet aura",
  "profile": "product",
  "num_images": 3,
  "aspect_ratio": "card_portrait",
  "negative_prompt": "blurry, watermark",
  "seed": 1000,
  "spread": 50,
  "style_preset": "dark_fantasy",
  "lora_strength": 0.8,
  "upscale_quality": "enhanced",
  "refine": false
}
```

```json
{
  "images": [
    { "image": "<base64 png>", "seed": 1043, "quality": [ { "gate": "tiling", "score": 0.04, "passed": true } ] }
  ],
  "refined_prompt": null
}
```

| Field | Type | Notes |
|---|---|---|
| `profile` | enum | `character` · `product` · `scene_frame` — **required** |
| `prompt` | string | ≤ 600 chars — **required** |
| `num_images` | int | 1–10 — **required** |
| `negative_prompt` | string? | ≤ 600 chars |
| `subject` / `environment` / `feeling` | string? | Prompt-refinement context |
| `refine` | bool | Default `false`. See [Prompt refinement](#prompt-refinement) |
| `seed` / `spread` | int? | See [Seeds and batches](#seeds-and-batches) |
| `aspect_ratio` | enum? | Default `square`. Invalid values coerce to `square` |
| `controls` | object? | ControlNet images + per-control type and strength, max 3 |
| `style_preset` | enum? | 8 LoRA presets |
| `lora_strength` | float? | 0.0–1.0 |
| `upscale_quality` | enum? | `none` · `enhanced` (ESRGAN) · `generative` (latent x4) |

`GET /health` returns `{"status": "ok"}` — liveness only; it does not indicate that any
model is loaded.

### Seeds and batches

Verified against the code. Three cases:

| Request | Behaviour |
|---|---|
| **`seed` + `spread`** | Each image draws from `[seed-spread, seed+spread]`. Deliberately non-deterministic — the same request twice gives different seeds. Each seed **is** recorded in the response and the filename |
| **`seed`, no `spread`** | ⚠️ Every image in the batch gets the *same* seed, and all write to the same `seed_<n>.png` — only the last survives on disk. The response still returns all N images. **Known open issue** |
| **No `seed`** | Unseeded generator per image. Files get a positional suffix (`seed_NaN_1.png`) so they don't collide, but the actual value is not captured — `seed` is `null` in the response and these images cannot be reproduced |

**Fully reproducible today:** a single image with an explicit `seed`.

---

## Prompt refinement

With `refine: true`, the prompt is expanded into a detailed SDXL-oriented description
before generation. The path is **hybrid, not purely local**:

1. **Groq** `llama-3.3-70b-versatile` — cloud API, requires `GROQ_API_KEY`
2. **Ollama** local Mistral 7B — if Groq fails or the key is unset
3. **Passthrough** — the original prompt, unchanged, if both fail

Setting `refine: true` with a Groq key configured **sends your prompt to a third-party
cloud API.** For fully offline operation, leave `GROQ_API_KEY` unset (Ollama-only) or
use `refine: false`. Generation itself — the diffusion pipeline — is always local.

---

## Requirements

- Python 3.10+
- CUDA GPU — developed and tested on RTX 5070 Ti, 16 GB, Blackwell / cu128
- [Ollama](https://ollama.com/) — optional, local refinement fallback
- [Groq](https://groq.com/) API key — optional, primary refinement path

## Installation

```bash
git clone https://github.com/chripara/img-engine.git
cd img-engine
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt
python run.py
```

Gradio UI at `http://127.0.0.1:7860`, API at `http://127.0.0.1:5000`.

Checkpoints, ControlNets, and LoRAs are pulled from HuggingFace on first use. ESRGAN
weights go in `local_models/`.

| Env var | Purpose |
|---|---|
| `GROQ_API_KEY` | Enables the cloud refinement path. Optional |
| `OLLAMA_PATH` | Overrides Ollama auto-detection. Optional — the app starts fine without Ollama |

---

## Limitations

Real and current, not aspirational.

- **No automated tests.** Every issue below was found by reading code, not by a failing
  test. This is the top priority.
- **No performance numbers.** Latency, peak VRAM per backend combination, and steps-vs-quality
  are unmeasured. Settings in the profile registry are informed defaults, not benchmark results.
- **Quality gates are 1 of 5.** Tiling detection works; `hands`, `face`, `clip`, and `iqa`
  are declared in the enum with thresholds but have no implementation. Existing thresholds
  are unvalidated against labelled data, and gate results are advisory — nothing acts on them.
- **Batch seed collision** when `seed` is given without `spread` — see above.
- **No concurrency control.** Two simultaneous requests instantiate two pipelines and
  can exhaust VRAM. Effectively single-user.
- **Synchronous base64 responses.** A batch of 10 at 1024² is ~15–20 MB in one JSON body,
  with the request blocking for minutes. Fine for local use, wrong shape for a service.
- **Weights are unpinned.** HuggingFace repo IDs without revisions — upstream updates can
  change output for the same inputs.
- **No Docker, no CI, no deployment.** Runs from source.

---

## Roadmap

- [x] **E01** — Local SDXL-class generation
- [x] **E02** — Inputs, batching, long-prompt chunking
- [x] **E03** — ControlNet, aspect ratios, negatives
- [x] **E04** — Non-functional requirements
- [ ] **E05** — Constraints & interface
- [x] **E06** — LoRA and style presets
- [ ] **E07** — Contract unification
- [ ] **E08** — Quality gates and profile benchmarking
- [ ] **E09** — Output pipeline stages

**Next up:** a labelled evaluation set, the four missing gates with thresholds calibrated
against it, and a checkpoint × profile benchmark to replace the current registry defaults
with measured ones.

---

## Stack

Diffusers · Compel · PyTorch/CUDA · Flask · Pydantic v2 · Gradio · spandrel (ESRGAN) ·
Groq + Ollama (optional refinement)

---

## License

MIT — see [LICENSE](LICENSE).

Model weights are **not** covered by this license. Checkpoints, ControlNets, LoRAs, and
upscaler weights are downloaded at runtime and carry their own terms — SDXL checkpoints
ship under CreativeML Open RAIL++-M, which imposes use restrictions. Review the license
of any model before using generated output commercially.