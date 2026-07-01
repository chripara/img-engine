from __future__ import annotations
import io, torch, hashlib, time, os, gc
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionUpscalePipeline
from app.services.upscaler.registries.upscaler_registry import _UPSCALERS
from app.services.registries.profile_registry import Profile, _PROFILES
from app.schemas.generate import GenerateRequest
from app.services.upscaler.backends.base_backend import BaseBackend

from utils.enums import Upscaler

class LatentDiffusionBackend(BaseBackend):
    def __init__(self, denoising_strength: float = 0.3) -> None:
        self._model_path = _UPSCALERS[Upscaler.LATENT]
        self._denoising_strength = denoising_strength
        self._pipe: StableDiffusionUpscalePipeline | None = None

    def load(self) -> None:
        self._pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
        ).to("cuda")

    def upscale(self, image: Image.Image, req: GenerateRequest) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("LatentDiffusionBackend not loaded. Call load() first.")

        tile_size = 512
        overlap = 64

        w, h = image.size
        result = Image.new("RGB", (w * 4, h * 4))

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                tile = image.crop((x, y, min(x + tile_size, w), min(y + tile_size, h)))
                upscaled_tile = self._pipe(
                    prompt=req.prompt,
                    image=tile,
                    noise_level=int(self._denoising_strength * 100),
                    num_inference_steps=8,
                ).images[0]
                result.paste(upscaled_tile, (x * 4, y * 4))

        image = result
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95, dpi=(300, 300))
        png_bytes = buffer.getvalue()

        filename = f"seed_{req.seed if req.seed else 'NaN'}_latent.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)

        return image

    def unload(self) -> None:
        self._pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def __enter__(self) -> LatentDiffusionBackend:
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()