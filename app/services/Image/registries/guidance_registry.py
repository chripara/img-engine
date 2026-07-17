from typing import Mapping

from utils.enums import Checkpoint, GuidanceType, Profile
from diffusers import ControlNetModel, SD3ControlNetModel, FluxControlNetModel

_SDXL_CONTROLNET_MODELS: dict[GuidanceType, str] = {
    GuidanceType.CANNY:    "diffusers/controlnet-canny-sdxl-1.0",
    GuidanceType.DEPTH:    "diffusers/controlnet-depth-sdxl-1.0",
    GuidanceType.POSE:     "thibaud/controlnet-openpose-sdxl-1.0",
    GuidanceType.SCRIBBLE: "xinsir/controlnet-scribble-sdxl-1.0",
}
# _GUIDANCE_MODELS: dict[Profile, type[ModelMixin]] = {
#     profile: ControlNetModel
#     for profile in (Profile.CHARACTER, Profile.SCENE_FRAME, Profile.PRODUCT)
# }

_GUIDANCE_MODELS: dict[Profile, type[ControlNetModel | SD3ControlNetModel | FluxControlNetModel]] = {
    Profile.CHARACTER: ControlNetModel,
    Profile.PRODUCT: ControlNetModel,
    Profile.SCENE_FRAME: ControlNetModel,
}