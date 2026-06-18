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