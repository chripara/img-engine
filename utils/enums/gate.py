from enum import Enum

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
