from typing import Type
from dataclasses import dataclass
from utils.enums.checkpoint import Checkpoint
from utils.enums.profile import Profile
from utils.enums.upscale import Upscaler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

@dataclass
class ProfileSpec:
    name: str
    model: Checkpoint
    scheduler: Type        # class reference
    steps: int
    cfg: float
    native_size: tuple     # (width, height)
    esrgan_upscaler: Upscaler
    vae_id: str | None = None      # "madebyollin/sdxl-vae-fp16-fix" ή None

_PROFILES: dict[Profile, ProfileSpec] = {
    Profile.CHARACTER: ProfileSpec(
        name = Profile.CHARACTER.value,
        model = Checkpoint.ALBEDO_BASE,
        scheduler = EulerDiscreteScheduler,
        steps = 30,
        cfg = 7.0,
        native_size = (1024, 1024),
        esrgan_upscaler = Upscaler.ESRGAN,
        vae_id = None,
    ),
    Profile.PRODUCT: ProfileSpec(
        name = Profile.PRODUCT.value,
        model = Checkpoint.DREAMSHAPER_XL,
        scheduler = EulerDiscreteScheduler,
        steps = 30,
        cfg = 7.0,
        native_size = (1024, 1024),
        esrgan_upscaler = Upscaler.ANIME_ESRGAN,
        vae_id = "madebyollin/sdxl-vae-fp16-fix",
    ),
    Profile.SCENE_FRAME: ProfileSpec(
        name = Profile.SCENE_FRAME.value,
        model = Checkpoint.JUGGERNAUT_XL,
        scheduler = DPMSolverMultistepScheduler,
        steps = 35,
        cfg = 4.5,
        native_size = (832, 1216),
        esrgan_upscaler = Upscaler.ESRGAN,
        vae_id = None,
    ),
}