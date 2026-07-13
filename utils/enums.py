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

class UpscaleQuality(Enum):
    NONE = "none"
    ENHANCED = "enhanced"
    GENERATIVE = "generative"

class Upscaler(Enum):
    ESRGAN = "esrgan"
    ANIME_ESRGAN = "anime_esrgan"
    LATENT = "latent"

class GuidanceType(str, Enum):
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    SCRIBBLE = "scribble"

class AspectRatio(str, Enum):
    SQUARE        = "square"         # 1024×1024
    LANDSCAPE     = "landscape"      # 1344×768
    PORTRAIT      = "portrait"       # 768×1344
    CARD_PORTRAIT = "card_portrait"  # 832×1216
    CARD_LARGE    = "card_large"     # 1152×896

class GateType(str, Enum):
    TILING = "tiling"
    HANDS  = "hands"
    FACE   = "face"
    CLIP   = "clip"
    IQA    = "iqa"

class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"