from enum import Enum

class ImgBackend(Enum):
    SDXL = "sdxl"
    FLUX = "flux"
    API  = "api"

class Checkpoint(Enum):
    SDXL_BASE = "sdxlbase"
    ALBEDO_BASE = "albedo_base"
    JUGGERNAUT_XL = "juggernaut_xl"
    DREAMSHAPER_XL = "dreamshaper_xl"

class ModelSource(Enum):
    DIFFUSERS = "diffusers"        
    SINGLE_FILE = "single_file"    
    GGUF = "gguf"                  

class Profile(Enum):
    CHARACTER = "character"
    PRODUCT = "product"
    SCENE_FRAME = "scene_frame"

class Resolution(Enum):
    STANDARD = "standard 1024"
    HIGH = "high 2K" 
    ULTRA = "ultra 4K+"

class UpscaleQuality(Enum):
    NONE = "none"
    ENHANCED = "enhanced"
    GENERATIVE = "generative"

class Upscaler(Enum):
    ESRGAN = "esrgan"
    ANIME_ESRGAN = "anime_esrgan"
    LATENT = "latent"