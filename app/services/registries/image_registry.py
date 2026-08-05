from pydantic import BaseModel

from utils.enums.aspect_ratio import AspectRatio

class Dimensions(BaseModel):
    width: int
    height: int

_ASPECT_RATIOS: dict[AspectRatio, Dimensions] = {
    AspectRatio.SQUARE:        Dimensions(width = 1024, height = 1024),
    AspectRatio.LANDSCAPE:     Dimensions(width = 1344, height = 768),
    AspectRatio.PORTRAIT:      Dimensions(width = 768, height = 1344),
    AspectRatio.CARD_PORTRAIT: Dimensions(width = 832, height = 1216),
    AspectRatio.CARD_LARGE:    Dimensions(width = 1152, height = 896),
}

_SDXL_CONTROLNET_LIMIT: int = 3