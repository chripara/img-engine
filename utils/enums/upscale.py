from enum import Enum

class UpscaleQuality(Enum):
    NONE = "none"
    ENHANCED = "enhanced"
    GENERATIVE = "generative"

class Upscaler(Enum):
    ESRGAN = "esrgan"
    ANIME_ESRGAN = "anime_esrgan"
    LATENT = "latent"
