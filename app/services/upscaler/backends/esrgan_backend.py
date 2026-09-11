import io, os
from PIL import Image

from app.services.upscaler.backends.base_backend import BaseBackend
from app.services.upscaler.backends.tensor_upscale_runner import TensorUpscaleRunner
from app.services.upscaler.backends.spandrel_tensor_runner import SpandrelTensorRunner
from app.services.upscaler.registries.upscaler_registry import _UPSCALERS
from app.schemas.generate import GenerateRequest
from utils.enums.upscale import Upscaler


class ESRGANBackend(BaseBackend):
    def __init__(self, upscaler: Upscaler, runner: TensorUpscaleRunner | None = None) -> None:
        self._model_path = _UPSCALERS[upscaler]
        self._runner: TensorUpscaleRunner = runner if runner is not None else SpandrelTensorRunner()

    def load(self) -> None:
        self._runner.load(self._model_path)

    def upscale(self, image: Image.Image, req: GenerateRequest, index: int = 0, seed: int | None = None) -> Image.Image:
        result = self._runner.run(image)
        self._write_debug_png(result, seed, index)
        return result

    def unload(self) -> None:
        self._runner.unload()

    def _write_debug_png(self, image: Image.Image, seed: int | None, index: int) -> None:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95, dpi=(300, 300))
        png_bytes = buffer.getvalue()

        filename = f"seed_{seed if seed is not None else f'NaN_{index + 1}'}_esrgan.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)
