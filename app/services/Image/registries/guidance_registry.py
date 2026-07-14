from utils.enums import Checkpoint, GuidanceType

_SDXL_CONTROLNET_MODELS: dict[GuidanceType, str] = {
    GuidanceType.CANNY:    "diffusers/controlnet-canny-sdxl-1.0",
    GuidanceType.DEPTH:    "diffusers/controlnet-depth-sdxl-1.0",
    GuidanceType.POSE:     "thibaud/controlnet-openpose-sdxl-1.0",
    GuidanceType.SCRIBBLE: "xinsir/controlnet-scribble-sdxl-1.0",
}

