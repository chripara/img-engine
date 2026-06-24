from utils.enums import Checkpoint

_CHECKPOINT: dict[Checkpoint, str] = {
    Checkpoint.SDXL_BASE: "stabilityai/stable-diffusion-xl-base-1.0",
    Checkpoint.ALBEDO_BASE: "stablediffusionapi/albedobase-xl",
    Checkpoint.JUGGERNAUT_XL: "RunDiffusion/Juggernaut-XL-v9", #r"C:\Projects\TextToVideo\Models\For Image\SDXL\juggernautXL_ragnarok.safetensors",
    Checkpoint.DREAMSHAPER_XL: "Lykon/dreamshaper-xl-1-0"
}