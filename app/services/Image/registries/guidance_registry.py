from utils.enums.guidance import GuidanceType
from utils.enums.profile import Profile
from diffusers import ControlNetModel, SD3ControlNetModel, FluxControlNetModel

_SDXL_CONTROLNET_MODELS: dict[GuidanceType, str] = {
    GuidanceType.CANNY:    "diffusers/controlnet-canny-sdxl-1.0",
    GuidanceType.DEPTH:    "diffusers/controlnet-depth-sdxl-1.0",
    GuidanceType.POSE:     "thibaud/controlnet-openpose-sdxl-1.0",
    GuidanceType.SCRIBBLE: "xinsir/controlnet-scribble-sdxl-1.0",
}

_GUIDANCE_MODELS: dict[Profile, type[ControlNetModel | SD3ControlNetModel | FluxControlNetModel]] = {
    Profile.CHARACTER: ControlNetModel,
    Profile.PRODUCT: ControlNetModel,
    Profile.SCENE_FRAME: ControlNetModel,
}