import io, os
from PIL import Image

from app.schemas.generate import GuidanceResult
from app.services.image.backends.base_backend import BaseBackend
from app.services.image.backends.model_runner import ImageModelRunner
from app.services.image.backends.sdxl_model_runner import SDXLModelRunner
from app.services.image.registries.checkpoint_registry import _CHECKPOINT
from app.services.image.registries.guidance_registry import _GUIDANCE_MODELS, _SDXL_CONTROLNET_MODELS
from app.services.image.registries.stype_presets import _STYLE_PRESET_REGISTRY
from app.services.registries.image_registry import Dimensions, _SDXL_CONTROLNET_LIMIT
from app.services.registries.profile_registry import _PROFILES
from utils.enums.guidance import GuidanceType
from utils.enums.profile import Profile
from utils.enums.style_presets import StylePreset


class SDXLBackend(BaseBackend):
    def __init__(self, profile: Profile, runner: ImageModelRunner | None = None) -> None:
        self._steps = _PROFILES[profile].steps
        self._cfg = _PROFILES[profile].cfg
        self._runner: ImageModelRunner = runner if runner is not None else SDXLModelRunner()

    def load(self, profile: Profile, style_preset: StylePreset, lora_weight: float | None, use_controlnet: bool, guidance_types: list[GuidanceType]) -> None:
        controlnet_model_cls = None
        controlnet_repo_ids: list[str] = []
        if use_controlnet:
            controlnet_model_cls = _GUIDANCE_MODELS[profile]
            controlnet_repo_ids = [_SDXL_CONTROLNET_MODELS[gt] for gt in guidance_types]

        self._runner.load(
            checkpoint_id=_CHECKPOINT[_PROFILES[profile].model],
            vae_id=_PROFILES[profile].vae_id,
            controlnet_model_cls=controlnet_model_cls,
            controlnet_repo_ids=controlnet_repo_ids,
            use_cpu_offload=len(guidance_types) >= _SDXL_CONTROLNET_LIMIT,
        )

        if style_preset is not None:
            strength = lora_weight if lora_weight is not None else 0.8
            try:
                self._runner.apply_lora(_STYLE_PRESET_REGISTRY[style_preset], style_preset.value, strength)
            except Exception as e:
                print(f"LoRA loading failed for preset '{style_preset.value}': {e}. Continuing without style preset.")

        self._runner.bind_scheduler(_PROFILES[profile].scheduler)

    def generate(self, prompt: str, negative_prompt: str | None, dimensions: Dimensions, seed: int | None, controls: list[GuidanceResult] | None, index: int = 0) -> Image.Image:
        control_images = [c.image for c in controls] if controls is not None else None
        control_strengths = [c.strength for c in controls if c.strength is not None] if controls is not None else None

        image = self._runner.run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=dimensions.width,
            height=dimensions.height,
            steps=self._steps,
            cfg=self._cfg,
            seed=seed,
            control_images=control_images,
            control_strengths=control_strengths,
            index=index,
        )

        self._write_debug_png(image, seed, index)
        return image

    def unload(self) -> None:
        self._runner.unload()

    def _write_debug_png(self, image: Image.Image, seed: int | None, index: int) -> None:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95, dpi=(300, 300))
        png_bytes = buffer.getvalue()

        filename = f"seed_{seed}.png" if seed is not None else f"seed_NaN_{index + 1}.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)