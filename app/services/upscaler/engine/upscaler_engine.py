import torch, gc
from typing import Callable
from PIL import Image
from utils.enums import UpscaleQuality, Resolution, Upscaler
from app.schemas.generate import GenerateRequest
from app.services.registries.profile_registry import ProfileSpec, _PROFILES
from app.services.upscaler.backends.esrgan_backend import ESRGANBackend
from app.services.upscaler.backends.latent_diffusion_backend import LatentDiffusionBackend
from app.services.upscaler.registries.upscaler_registry import _UPSCALERS
from app.services.upscaler.registries.backend_registry import _BACKENDS

DENOISINT_STRENGTH = 0.3

class UpscalerEngine:
    def __init__(self, req: GenerateRequest, spec: ProfileSpec):
        self._upscaler = self._get_backend(req,spec)    

    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        del self._upscaler
        torch.cuda.empty_cache()
        gc.collect()

    def _get_backend(self, 
        req: GenerateRequest,
        spec: ProfileSpec,
    ) -> ESRGANBackend | LatentDiffusionBackend | None:
        match req.upscale_quality:
            case UpscaleQuality.NONE:
                return None
            case UpscaleQuality.ENHANCED:
                return ESRGANBackend(upscaler=spec.esrgan_upscaler)
            case UpscaleQuality.GENERATIVE:
                return LatentDiffusionBackend(
                    denoising_strength = DENOISINT_STRENGTH,
                )
            
    def upscale_image(self, img: Image.Image, req: GenerateRequest) -> Image.Image:
        if self._upscaler is None:
            return img

        self._upscaler.load()
        result = self._upscaler.upscale(img, req)

        self._upscaler.unload()
        return result