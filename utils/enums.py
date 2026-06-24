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
    DIFFUSERS = "diffusers"        # from_pretrained()
    SINGLE_FILE = "single_file"    # from_single_file()
    GGUF = "gguf"                  # from_single_file() with special config

class Profile(Enum):
    CHARACTER = "character"
    PRODUCT = "product"
    SCENE_FRAME = "scene_frame"
