from enum import Enum

class GuidanceType(str, Enum):
    CANNY = "canny"
    DEPTH = "depth"
    POSE = "pose"
    SCRIBBLE = "scribble"
