from typing import Type
from dataclasses import dataclass
from utils.enums import Checkpoint, Profile
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler

@dataclass
class ProfileSpec:
    name: str
    description: str | None
    model: Checkpoint
    scheduler: Type        # class reference
    steps: int
    cfg: float
    native_size: tuple     # (width, height)
    default_negative: str  # "" μέχρι E08-S05
    vae_id: str | None = None      # "madebyollin/sdxl-vae-fp16-fix" ή None
    refiner: bool = False
    upscale: str = "none"  # none | 1440p | 4k
    restore: bool = False
    isolate: bool = False
    variant: str | None = None

_PROFILES: dict[Profile, ProfileSpec] = {
    Profile.CHARACTER: ProfileSpec(
        name = Profile.CHARACTER.value,
        description = Profile.CHARACTER.value,
        model = Checkpoint.ALBEDO_BASE,
        scheduler = EulerDiscreteScheduler,
        steps = 30,
        cfg = 7.0,
        native_size = (1024, 1024),
        default_negative = "",
        vae_id = None,
        refiner = False,
        upscale = "none",
        restore = False,
        isolate = False,
        variant = None,
    ),
    Profile.PRODUCT: ProfileSpec(
        name = Profile.PRODUCT.value,
        description = "Equipment, weapons, relics, icons — isolated objects",
        model = Checkpoint.DREAMSHAPER_XL,
        scheduler = DPMSolverSinglestepScheduler,
        steps = 30,
        cfg = 7.0,
        native_size = (1024, 1024),
        default_negative = "",
        vae_id = "madebyollin/sdxl-vae-fp16-fix",
        refiner = False,
        upscale = "none",
        restore = False,
        isolate = True,
        variant = None,
    ),
    Profile.SCENE_FRAME: ProfileSpec(
        name = Profile.SCENE_FRAME.value,
        description = "Card frames, backgrounds, environments, logo",
        model = Checkpoint.JUGGERNAUT_XL,
        scheduler = DPMSolverMultistepScheduler,
        steps = 35,
        cfg = 4.5,
        native_size = (832, 1216),
        default_negative = "",
        vae_id = None,
        refiner = False,
        upscale = "none",
        restore = False,
        isolate = False,
        variant = "fp16",
    ),
}