from dataclasses import dataclass
from utils.enums import Upscaler

_UPSCALERS: dict[Upscaler, str] = {
    Upscaler.ESRGAN: "local_models\RealESRGAN_x4plus.pth",
    Upscaler.ANIME_ESRGAN: "local_models\RealESRGAN_x4plus_anime_6B.pth",
    Upscaler.LATENT: "stabilityai/stable-diffusion-x4-upscaler"
}
