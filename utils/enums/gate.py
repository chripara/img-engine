from enum import Enum

class GateType(str, Enum):
    TILING = "tiling"
    HANDS  = "hands"
    FACE   = "face"
    CLIP   = "clip"
    IQA    = "iqa"

class GateStatus(str, Enum):
    NOT_APPLICABLE = "not applicable"
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
