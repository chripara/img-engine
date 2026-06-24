from utils.enums import Checkpoint

_VAE: dict[Checkpoint, str | None] = {
    Checkpoint.SDXL_BASE: "madebyollin/sdxl-vae-fp16-fix",
    Checkpoint.ALBEDO_BASE: None,
    Checkpoint.JUGGERNAUT_XL: None, 
    Checkpoint.DREAMSHAPER_XL: "madebyollin/sdxl-vae-fp16-fix"
}