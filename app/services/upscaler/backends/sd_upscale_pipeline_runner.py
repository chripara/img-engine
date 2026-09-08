import torch, gc
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionUpscalePipeline

from app.services.upscaler.backends.tiled_diffusion_upscale_runner import TiledDiffusionUpscaleRunner


class SDUpscalePipelineRunner(TiledDiffusionUpscaleRunner):
    def __init__(self) -> None:
        self._pipe: StableDiffusionUpscalePipeline | None = None

    def load(self, model_path: str) -> None:
        self._pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
        self._pipe.enable_model_cpu_offload()

    def run_tile(self, prompt: str, tile: Image.Image, noise_level: int, steps: int) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("SDUpscalePipelineRunner not loaded. Call load() first.")

        return self._pipe(
            prompt=prompt,
            image=tile,
            noise_level=noise_level,
            num_inference_steps=steps,
        ).images[0]

    def unload(self) -> None:
        if self._pipe is not None:
            self._pipe.to("cpu")
        self._pipe = None
        torch.cuda.empty_cache()
        gc.collect()