# Architecture notes

Design decisions that complement the diagram in the [README](../README.md#architecture).

**Note on the Gradio UI:** the UI builds its own request object from raw form values and sends it as JSON over HTTP to the Flask API — it does not duplicate or bypass validation. The Flask API's `GenerateRequest` schema is the single source of truth; the UI's local schema is intentionally independent since it exists to serialize dropdown/form values for local debugging, not to enforce the contract.

**Note on the upscaler backends:** `ESRGANBackend` and `LatentDiffusionBackend` share their `load`/`unload` context-manager protocol via a common `BaseBackend.__enter__`/`__exit__`, so each backend loads its model exactly once per batch. Per-image seed and index are threaded through to the upscaler stage, so upscaled output filenames don't collide even across a seedless batch.

**Note on load/unload lifecycle:** the SDXL backend is loaded fresh and unloaded after every request rather than kept resident in VRAM between requests. This is a deliberate choice, not an oversight: measured `load()` cost is ~0.5–1s (`safetensors` uses memory-mapped loading, so most of that time is the `.to('cuda')` transfer and LoRA fuse, not disk I/O) — negligible next to actual generation time. An instance cache to avoid reloading between requests was evaluated and not pursued for that reason; the per-request load/unload also keeps VRAM usage predictable on single-GPU hardware.
