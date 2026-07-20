from enum import Enum

class AspectRatio(str, Enum):
    SQUARE        = "square"         # 1024×1024
    LANDSCAPE     = "landscape"      # 1344×768
    PORTRAIT      = "portrait"       # 768×1344
    CARD_PORTRAIT = "card_portrait"  # 832×1216
    CARD_LARGE    = "card_large"     # 1152×896
