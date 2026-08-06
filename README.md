# README — Changes Needed

Everything below is a documentation fix, not a code fix — the code stays as-is (or already got fixed elsewhere in the checklist); the README needs to catch up to what's actually built.

---

- [ ] **Profile → checkpoint table is wrong.** README currently shows DreamShaper XL for all 3 profiles. Actual `profile_registry.py`: `CHARACTER → ALBEDO_BASE`, `PRODUCT → DREAMSHAPER_XL`, `SCENE_FRAME → JUGGERNAUT_XL` — three different checkpoints. Update the table to match.

- [ ] **SCENE_FRAME native size is reversed.** README says "1216×832"; `profile_registry.py` has `native_size = (832, 1216)` (dataclass comment says the tuple is `(width, height)`, so that's 832 wide × 1216 tall — portrait). Confirm which orientation is actually correct and fix whichever side is wrong (README or the registry comment/values).

- [ ] **Roadmap checklist is stale.** Currently only shows E01 as done. `E02-S05` (long prompts / dual-CLIP chunking) and `E06` (LoRA + style presets) are already implemented per the backlog docs — check them off.

- [ ] **No `LICENSE` file exists**, despite the badge and footer both claiming MIT. Either add the actual `LICENSE` file or stop claiming a license you haven't published.

- [ ] **Undisclosed Groq cloud dependency.** README says "100% local & offline — no API keys, no cloud inference, full data privacy." In reality, `prompt_service.py`'s `refine_prompt_with_llama()` calls the **Groq cloud API** (`llama-3.3-70b-versatile`) as the **first**, default prompt-refinement path when `refine: true` — Ollama/Mistral is only the fallback if Groq fails. This is a real contradiction, not just a stale doc. Options:
  - Document the true hybrid behavior (Groq first, Ollama fallback) and drop/qualify the "100% local" claim when `refine: true`, **or**
  - Change the code so local-only is the actual default and Groq is an explicit opt-in — then the README claim becomes true again.
  - Either way, document the `GROQ_API_KEY` env var requirement for the Groq path.

- [ ] **`seed` + `spread` behavior needs its own section — not in the original SRS at all.** Discovered while fixing batch seed reproducibility: `spread` is a field that doesn't exist in the canonical SRS request schema (§11.2) — it's an extra, project-specific feature layered on top. Document the actual, intentional behavior:
  - `seed` given, **no** `spread`, `batch_count = N` → deterministic: images get `seed, seed+1, seed+2, ...` (per SRS FR-IMG-024). Same request twice → identical results.
  - `seed` + `spread` given → **intentionally non-deterministic**: each image gets a random value in `[seed - spread, seed + spread]`, freshly randomized on every call. This is a deliberate exploration feature (get creative variations near a seed), not a bug — same request run twice will **not** give the same images. Worth being explicit about this so nobody reports it as a reproducibility bug later.
  - No `seed` at all → fully random seeds, one per image, each recorded in the response/filenames.

- [ ] **`OLLAMA_PATH` env var** — now that bug #1 is fixed (env var + `shutil.which` auto-detect + try/except instead of a hardcoded Windows path), document that `OLLAMA_PATH` can be set to override auto-detection, and that the app now starts fine even if Ollama isn't installed (PRE just falls back to Groq-only or passes the prompt through unrefined).

- [ ] **`ui/schema.py` vs `app/schemas/generate.py` — intentional split, not a bug.** The Gradio UI is a dev-only convenience tool (SRS E05-S02: "Any HTTP service or Gradio UI is dev-only and out of contract"), not part of the contract. Its own, looser schema (e.g. `profile: str` instead of the `Profile` enum) exists because it works with raw values straight from dropdowns, not because it's trying to mirror the API's validation. Only `app/schemas/generate.py` is the source of truth for validation — document this explicitly so the two schemas aren't mistaken for drift/inconsistency by a future reader.
