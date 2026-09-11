import io, os
from PIL import Image

from app.services.upscaler.backends.base_backend import BaseBackend
from app.services.upscaler.backends.tiled_diffusion_upscale_runner import TiledDiffusionUpscaleRunner
from app.services.upscaler.backends.sd_upscale_pipeline_runner import SDUpscalePipelineRunner
from app.services.upscaler.registries.upscaler_registry import _UPSCALERS
from app.schemas.generate import GenerateRequest
from utils.enums.upscale import Upscaler


class LatentDiffusionBackend(BaseBackend):
    def __init__(self, denoising_strength: float = 0.3, runner: TiledDiffusionUpscaleRunner | None = None) -> None:
        self._model_path = _UPSCALERS[Upscaler.LATENT]
        self._denoising_strength = denoising_strength
        self._runner: TiledDiffusionUpscaleRunner = runner if runner is not None else SDUpscalePipelineRunner()

    def load(self) -> None:
        self._runner.load(self._model_path)

    def upscale(self, image: Image.Image, req: GenerateRequest, index: int = 0, seed: int | None = None) -> Image.Image:
        w, h = image.size
        tile_size_x = w // 2
        tile_size_y = h // 2
        overlap_x = w // 16
        overlap_y = h // 16
        result = Image.new("RGB", (w * 4, h * 4))

        for y in range(0, h, tile_size_y - overlap_y):
            for x in range(0, w, tile_size_x - overlap_x):
                tile = image.crop((x, y, min(x + tile_size_x, w), min(y + tile_size_y, h)))
                upscaled_tile = self._runner.run_tile(
                    prompt=req.prompt,
                    tile=tile,
                    noise_level=int(self._denoising_strength * 100),
                    steps=8,
                )
                result.paste(upscaled_tile, (x * 4, y * 4))

        self._write_debug_png(result, seed, index)
        return result

    def unload(self) -> None:
        self._runner.unload()

    def _write_debug_png(self, image: Image.Image, seed: int | None, index: int) -> None:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95, dpi=(300, 300))
        png_bytes = buffer.getvalue()

        filename = f"seed_{seed if seed is not None else f'NaN_{index + 1}'}_latent.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)
