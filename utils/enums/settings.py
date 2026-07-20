from enum import Enum

class ImgBackend(Enum):
    SDXL = "sdxl"
    FLUX = "flux"
    API  = "api"

class ModelSource(Enum):
    DIFFUSERS = "diffusers"        
    SINGLE_FILE = "single_file"    
    GGUF = "gguf"
