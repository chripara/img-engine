from __future__ import annotations
import io, torch, os, gc
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionUpscalePipeline
from app.services.upscaler.registries.upscaler_registry import _UPSCALERS
from app.services.registries.image_registry import _ASPECT_RATIOS
from app.schemas.generate import GenerateRequest
from app.services.upscaler.backends.base_backend import BaseBackend
from utils.enums.upscale import Upscaler
from utils.enums.aspect_ratio import AspectRatio


class LatentDiffusionBackend(BaseBackend):
    def __init__(self, denoising_strength: float = 0.3) -> None:
        self._model_path = _UPSCALERS[Upscaler.LATENT]
        self._denoising_strength = denoising_strength
        self._pipe: StableDiffusionUpscalePipeline | None = None

    def load(self) -> None:
        self._pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
        )
        self._pipe.enable_model_cpu_offload()

    def upscale(self, image: Image.Image, req: GenerateRequest, index: int = 0) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("LatentDiffusionBackend not loaded. Call load() first.")

        dimensions = _ASPECT_RATIOS[req.aspect_ratio] if req.aspect_ratio else _ASPECT_RATIOS[AspectRatio.SQUARE]

        tile_size_x = dimensions.width // 2
        tile_size_y = dimensions.height // 2

        overlap_x = dimensions.width // 16
        overlap_y = dimensions.height // 16

        w, h = image.size
        result = Image.new("RGB", (w * 4, h * 4))

        for y in range(0, h, tile_size_y - overlap_y):
            for x in range(0, w, tile_size_x - overlap_x):
                tile = image.crop((x, y, min(x + tile_size_x, w), min(y + tile_size_y, h)))
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

        filename = f"seed_{req.seed if req.seed else 'NaN'}_latent_{index + 1}.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)

        return image

    def unload(self) -> None:
        if self._pipe is not None:
            self._pipe.to("cpu")
        self._pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        
    def __enter__(self) -> LatentDiffusionBackend:
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()