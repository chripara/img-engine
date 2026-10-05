# Setup & Instructions

Full reference for running img-engine locally and calling its API. See the [main README](../README.md) for architecture, quality gates, and known limitations.

---

## Requirements

- Python 3.11+
- CUDA-capable GPU (tested on RTX 5070 Ti, 16 GB VRAM, Blackwell/cu128)
- [Ollama](https://ollama.com/) (optional — local fallback for prompt refinement; app starts fine without it)
- A [Groq](https://groq.com/) API key (optional — only needed for the default, higher-quality PRE path; see [Prompt Refinement Engine](../README.md#prompt-refinement-engine-pre))

---

## Installation

```bash
git clone https://github.com/chripara/img-engine.git
cd img-engine
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

The `+cu128` PyTorch build pinned in `requirements.txt` is not on PyPI, which is why the extra index URL is needed.

**Models.** The SDXL checkpoints (see `app/services/image/registries/checkpoint_registry.py`), the VAE, the ControlNet annotators and the quality-gate models are Hugging Face repo IDs. They are downloaded automatically on first use and cached under your user profile, so the first run needs an internet connection and several GB of disk.

The only weights you must provide yourself are the two ESRGAN upscaler files, read from `local_models/` (git-ignored): `RealESRGAN_x4plus.pth` and `RealESRGAN_x4plus_anime_6B.pth`.

**Use a dedicated virtual environment for this project.** Installing into a shared/global Python environment risks dependency conflicts with other projects (observed in practice: a global `torch`/`diffusers`/`mediapipe`/`protobuf` version clash caused by unrelated projects sharing the same site-packages). Always `venv\Scripts\activate` before running or installing anything for img-engine.

---

## Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Optional | Enables the primary (cloud) PRE path. Without it, PRE falls back to local Ollama/Mistral, or passes the prompt through unrefined if neither is available. |
| `OLLAMA_PATH` | Optional | Overrides auto-detection of the Ollama executable. If unset, the app tries `shutil.which("ollama")`, then falls back to assuming `ollama` is on PATH. |
| `APP_ENV` | Optional | Set to `development` to enable Flask debug mode. Unset (the default) runs with `DEBUG=False`. Do not enable it when the API is reachable beyond localhost (`run.py` binds to `0.0.0.0`). |

---

## Running the server

```bash
python run.py
```

**Gradio UI:** open `http://127.0.0.1:7860` in your browser. Dev/testing convenience only — not part of the API contract.

**Interactive API docs:** once the server is running, Swagger UI and ReDoc are available at `/docs/swagger` and `/docs/redoc` respectively (raw OpenAPI spec at `/docs/openapi.json`) — auto-generated from the `GenerateRequest` Pydantic schema via `flask_pydantic_spec`. This is the authoritative, always-in-sync API reference; the example below is a quick-look, not the source of truth.

---

## REST API

```bash
POST /generate
Content-Type: application/json

{
  "prompt": "tanzanite crystal orb held in an open palm, deep violet aura, fantasy game item",
  "profile": "product",
  "feeling": "Mystical & Ethereal",
  "environment": "Dark Dungeon",
  "num_images": 3,
  "refine": false
}
```

Response:
```json
{
  "images": [
    {"image": "<base64>", "seed": 12345, "quality": [ {"gate": "CLIP", "score": 0.34, "passed": true, "suggested": null}, ... ]}
  ],
  "refined_prompt": "..."
}
```
