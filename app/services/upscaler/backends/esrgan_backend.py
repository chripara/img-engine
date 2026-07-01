from __future__ import annotations

import numpy as np
import io, torch, hashlib, time, os, gc
from PIL import Image
from spandrel import ModelLoader
from app.services.upscaler.registries.upscaler_registry import _UPSCALERS
from app.services.upscaler.backends.base_backend import BaseBackend
from utils.enums import Upscaler
from app.schemas.generate import GenerateRequest

_ESRGAN_NUM_BLOCK: dict[Upscaler, int] ={
    Upscaler.ESRGAN: 23,
    Upscaler.ANIME_ESRGAN: 6
}

class ESRGANBackend(BaseBackend):
    def __init__(self, upscaler: Upscaler) -> None:
        self._model_path = _UPSCALERS[upscaler]
        self._model = None

    def load(self) -> None:
        self._model = ModelLoader().load_from_file(self._model_path).cuda()

    def upscale(self, image: Image.Image, req: GenerateRequest) -> Image.Image:
        self.load()
        if self._model is None:
            raise RuntimeError("ESRGANBackend not loaded. Call load() first.")
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to("cuda")

        with torch.no_grad():
            output = self._model(tensor)

        output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1)
        image = Image.fromarray((output * 255).byte().cpu().numpy())

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95, dpi=(300, 300))
        png_bytes = buffer.getvalue()

        hash = hashlib.md5(req.prompt.encode()).hexdigest()[:8]
        filename = f"{int(time.time())}_{hash}_esrgan.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)

        return image

    def unload(self) -> None:
        self._upsampler = None
        torch.cuda.empty_cache()
        gc.collect()

    def __enter__(self) -> ESRGANBackend:
        self.load()
        return self

    def __exit__(self, *_) -> None:
        self.unload()