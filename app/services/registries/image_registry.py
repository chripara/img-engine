from pydantic import BaseModel
from wcwidth import width

from utils.enums import AspectRatio

class Dimensions(BaseModel):
    width: int
    height: int

_ASPECT_RATIOS: dict[AspectRatio, Dimensions] = {
    AspectRatio.SQUARE:        Dimensions(1024, 1024),
    AspectRatio.LANDSCAPE:     Dimensions(1344, 768),
    AspectRatio.PORTRAIT:      Dimensions(768, 1344),
    AspectRatio.CARD_PORTRAIT: Dimensions(832, 1216),
    AspectRatio.CARD_LARGE:    Dimensions(1152, 896),
}