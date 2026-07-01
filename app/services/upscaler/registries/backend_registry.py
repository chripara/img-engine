from typing import Type

from app.services.upscaler.backends.base_backend import BaseBackend
from app.services.upscaler.backends.esrgan_backend import ESRGANBackend
from app.services.upscaler.backends.latent_diffusion_backend import LatentDiffusionBackend
from utils.enums import Upscaler

_BACKENDS: dict[Upscaler, Type[BaseBackend]] = {
    Upscaler.ESRGAN: ESRGANBackend,
    Upscaler.ANIME_ESRGAN: ESRGANBackend,
    Upscaler.LATENT: LatentDiffusionBackend
}