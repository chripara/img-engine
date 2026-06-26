# img-engine

> Local, offline SDXL-class image generation engine with profile-based model recipes, batch generation, prompt refinement, and VRAM-safe execution. Built for game asset pipelines and creative production workflows.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow?logo=huggingface)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What is img-engine?

**img-engine** is a fully local, offline image generation engine built on top of [Stable Diffusion XL (SDXL)](https://stability.ai/stable-diffusion) and the [HuggingFace Diffusers](https://github.com/huggingface/diffusers) library. It provides a structured, production-ready pipeline for generating high-quality images from text prompts — with no cloud dependency, no data leaving your machine.

Designed initially for generating game card artwork (characters, items, scene backgrounds), it is general-purpose and extensible for any creative or production use case requiring local AI image generation.

---

## Features

- **100% local & offline** — no API keys, no cloud inference, full data privacy
- **SDXL-class quality** — runs DreamShaper XL, Juggernaut XL, AlbedoBase XL, and SDXL base
- **Profile-based recipe registry** — each use case (Character, Product, Scene) binds to a complete generation recipe: checkpoint + VAE + scheduler + CFG + steps + native resolution
- **Compel integration** — bypasses the CLIP 77-token limit for long, detailed prompts (~150 tokens)
- **Batch generation** — generate 1–10 images per request with distinct seeds
- **Prompt Refinement Engine (PRE)** — optional Mistral 7B (via Ollama) rewrites short prompts into detailed image descriptions
- **Pydantic validation** — request schema enforced at the API boundary (prompt max 600 chars, batch N ∈ [1,10])
- **VRAM-safe execution** — context manager lifecycle: load → generate → unload + `torch.cuda.empty_cache()` per batch
- **Seed reproducibility** — explicit `torch.Generator` seeding for deterministic output
- **Gradio UI** — browser-based interface for local testing and exploration
- **Flask REST API** — JSON contract for pipeline integration

---

## Architecture

```
┌─────────────────────────────────┐
│         Gradio UI / CLI         │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│       Flask REST Controller     │  ← Pydantic validation (GenerateRequest)
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

---

## Profiles

| Profile | Use Case | Default Checkpoint | Native Size |
|---|---|---|---|
| `CHARACTER` | Hero / character art | DreamShaper XL | 1024×1024 |
| `PRODUCT` | Item / object hero shot | DreamShaper XL | 1024×1024 |
| `SCENE_FRAME` | Environment / background | DreamShaper XL | 1216×832 |

Each profile carries its own VAE, scheduler, CFG, steps, default negative prompt, and optional refiner — defined as data in the registry, not branching code.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 5070 Ti, 16 GB VRAM)
- [Ollama](https://ollama.com/) (optional, for prompt refinement with Mistral 7B)

---

## Installation

```bash
git clone https://github.com/chripara/img-engine.git
cd img-engine
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Download your preferred SDXL checkpoint (`.safetensors`) and set the path in `config.py`.

---

## Usage

### Start the server

```bash
python run.py
```

### Gradio UI

Open `http://127.0.0.1:7860` in your browser.

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

---

## Prompt Refinement Engine (PRE)

When `refine: true`, the engine sends the prompt to a local **Mistral 7B** model (via Ollama) before generation. The PRE expands short prompts into detailed image descriptions optimized for SDXL.

Ollama is managed automatically — started on app launch, stopped on exit.

---

## Roadmap

- [x] E01 — Local SDXL-class generation
- [ ] E02 — Inputs & batching
- [ ] E03 — Outputs (ControlNet, aspect ratios, negatives)
- [ ] E04 — Non-functional (VRAM, reproducibility)
- [ ] E05 — Constraints & interface
- [ ] E06 — LoRA & style presets
- [ ] E07 — Contract unification
- [ ] E08 — Quality & acceptance
- [ ] E09 — Output pipeline & quality stages

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference | Stable Diffusion XL (Diffusers) |
| Token handling | Compel |
| Prompt refinement | Mistral 7B via Ollama |
| API | Flask |
| UI | Gradio |
| Validation | Pydantic v2 |
| GPU | PyTorch + CUDA |

---

## License

MIT
