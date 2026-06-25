from utils.enums import Checkpoint


_CHECKPOINT: dict[Checkpoint, str] = {
    Checkpoint.SDXL_BASE: "stabilityai/stable-diffusion-xl-base-1.0",
    Checkpoint.ALBEDO_BASE: "stablediffusionapi/albedobase-xl",
    Checkpoint.JUGGERNAUT_XL: "stablediffusionapi/juggernaut-xl-v9",
    Checkpoint.DREAMSHAPER_XL: "Lykon/dreamshaper-xl-1-0"
}