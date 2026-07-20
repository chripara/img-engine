from utils.enums.style_presets import StylePreset

_STYLE_PRESET_REGISTRY: dict[StylePreset, str] = {
    StylePreset.FANTASY: "ntc-ai/SDXL-LoRA-slider.fantasy",
    StylePreset.DARK_FANTASY: "thwri/dark-gothic-fantasy-xl",
    StylePreset.CARTOONISH_FANTASY: "ntc-ai/SDXL-LoRA-slider.cartoon",
    StylePreset.CYBERPUNK: "jbilcke-hf/sdxl-cyberpunk-2077",
    StylePreset.REALISM_CARTOONISH: "ostris/photorealistic-slider-sdxl-lora",
    StylePreset.SCIFI_FANTASY: "e-n-v-y/envy-scifi-streamline-xl-01",
    StylePreset.MEDIEVAL_FANTASY: "thliang01/medieval-knight-sdxl-lora-v0-1",
    StylePreset.ANIME_AESTHETIC: "Linaqruf/pastel-anime-xl-lora",
}