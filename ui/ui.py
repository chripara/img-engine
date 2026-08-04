import base64
import io
import gradio as gr
import requests
from gradio_modal import Modal
from PIL import Image
from ui.schema import GenerateRequest, GuidanceInput, GuidanceSettings
from utils.enums.profile import Profile
from utils.enums.guidance import GuidanceType
from app.services.registries.profile_registry import _PROFILES
from app.services.registries.guidance_registry import _GUIDANCE_DETAILS
from utils.enums.style_presets import StylePreset

GUIDANCE_TYPES = [GuidanceType.CANNY, GuidanceType.DEPTH, GuidanceType.POSE, GuidanceType.SCRIBBLE]

CONTROLNET_CSS = """
#controlnet_modal {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1000;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    padding: 20px;
    max-width: 90vw;
    max-height: 85vh;
    overflow-y: auto;
}
"""

_STYLE_PRESET_LABELS: list[tuple[str, str | None]] = [
    ("None", None),
    ("Fantasy", StylePreset.FANTASY.value),
    ("Dark Fantasy", StylePreset.DARK_FANTASY.value),
    ("Cartoonish Fantasy", StylePreset.CARTOONISH_FANTASY.value),
    ("Cyberpunk", StylePreset.CYBERPUNK.value),
    ("Realism Cartoonish", StylePreset.REALISM_CARTOONISH.value),
    ("Technological Fantasy / Scifi Fantasy", StylePreset.SCIFI_FANTASY.value),
    ("Magic Medieval Fantasy", StylePreset.MEDIEVAL_FANTASY.value),
    ("Japan Anime Aesthetic", StylePreset.ANIME_AESTHETIC.value),
]

def _pil_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _update_strength_defaults(profile_value: str):
    if not profile_value:
        return [gr.update() for _ in GUIDANCE_TYPES]
    checkpoint = _PROFILES[Profile(profile_value)].model
    defaults = _GUIDANCE_DETAILS[checkpoint].defaults
    return [gr.update(value=defaults[gt]) for gt in GUIDANCE_TYPES]

def _update_selectors(canny_en, depth_en, pose_en, scribble_en):
    enabled_flags = [canny_en, depth_en, pose_en, scribble_en]
    selectors = []
    idx = 0
    for en in enabled_flags:
        if en:
            selectors.append(idx)
            idx += 1
        else:
            selectors.append(None)
    return selectors

def launch_ui():

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):

                with gr.Group():
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=3)
                    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here...", lines=1)
                    profile = gr.Dropdown(
                        label="Profile",
                        choices=[
                            ("Character / Hero", Profile.CHARACTER.value),
                            ("Product / Item", Profile.PRODUCT.value),
                            ("Scene / Card Frame", Profile.SCENE_FRAME.value),
                        ]
                    )
                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=[
                                ("Square 1024×1024", "square"),
                                ("Landscape 1344×768", "landscape"),
                                ("Portrait 768×1344", "portrait"),
                                ("Card Portrait 832×1216", "card_portrait"),
                                ("Card Large 1152×896", "card_large"),
                            ],
                            value="square",
                        )

                    gr.Markdown("### Style Preset")
                    with gr.Row():
                        style_preset = gr.Dropdown(
                            choices=_STYLE_PRESET_LABELS,
                            label="Style",
                            value=None,
                        )

                    gr.Markdown("### LoRA")
                    with gr.Row():
                        lora_strength = gr.Slider(0, 1, label="strength", value=0.5)

                with gr.Group():
                    with gr.Row():
                        feeling = gr.Dropdown(
                            label="Feeling",
                            choices=[
                                ("Dark & Gritty"),
                                ("Heroic & Epic"),
                                ("Mystical & Ethereal"),
                                ("Ancient & Forgotten"),
                                ("Chaotic & Wild"),
                                ("Serene & Hopeful"),
                                ("Ominous & Threatening"),
                                ("Vibrant & Energetic"),
                            ]
                        )
                        environment = gr.Dropdown(
                            label="Environment",
                            choices=[
                                ("Ancient Forest"),
                                ("Abandoned Ruins"),
                                ("Battlefield"),
                                ("Underground Cave"),
                                ("Mountaintop"),
                                ("Dark Dungeon"),
                                ("Sacred Temple"),
                                ("Open Sea"),
                                ("Volcanic Wasteland"),
                                ("Frozen Tundra"),
                            ]
                        )

                    with gr.Row():
                        subject = gr.Dropdown(
                            label="Subject",
                            choices=[
                                ("Person / Figure"),
                                ("Animal / Creature"),
                                ("Object / Item"),
                                ("Landscape / Scene"),
                                ("Building / Structure"),
                                ("Vehicle / Machine"),
                                ("Plant / Nature"),
                                ("Abstract / Concept"),
                                ("Group / Crowd"),
                                ("Event / Action"),
                            ]
                        )
                        refine = gr.Dropdown(
                            label="Refine",
                            choices=[
                                ("False", False),
                                ("True", True),
                            ]
                        )

                with gr.Group():
                    with gr.Row():
                        num_images = gr.Dropdown(
                            label="Number of Images",
                            choices=[(str(i), i) for i in range(1, 11)]
                        )
                        upscale_quality = gr.Dropdown(
                            label="Upscale Quality",
                            choices=[
                                ("none"),
                                ("enhanced"),
                                ("generative"),
                            ]
                        )

                with gr.Accordion("Seed", open=False):
                    with gr.Row(equal_height=True):
                        use_seed = gr.Checkbox(label="Use Seed", value=False)
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        spread = gr.Number(label="Spread", value=0, precision=0)

                with gr.Row():
                    controlnet_btn = gr.Button("Configure ControlNet")
                    generate_button = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Output Images")
                refined_prompt_box = gr.Textbox(label="Refined Prompt", interactive=False)
                quality_output = gr.JSON(label="Quality Gates")

        with gr.Group(visible=False, elem_id="controlnet_modal") as controlnet_modal:
            gr.Markdown("### ControlNet")
            with gr.Group():
                with gr.Row():
                    canny_en = gr.Checkbox(label="canny enable", value=False)
                    canny_img = gr.Image(label="canny Image", type="pil")
                    canny_sel = gr.Number(label="selector", value=None, interactive=False)
                    canny_str = gr.Slider(0, 1, label="strength", value=0.5)
            with gr.Group():
                with gr.Row():
                    depth_en = gr.Checkbox(label="depth enable", value=False)
                    depth_img = gr.Image(label="depth Image", type="pil")
                    depth_sel = gr.Number(label="selector", value=None, interactive=False)
                    depth_str = gr.Slider(0, 1, label="strength", value=0.5)
            with gr.Group():
                with gr.Row():
                    pose_en = gr.Checkbox(label="pose enable", value=False)
                    pose_img = gr.Image(label="pose Image", type="pil")
                    pose_sel = gr.Number(label="selector", value=None, interactive=False)
                    pose_str = gr.Slider(0, 1, label="strength", value=0.5)
            with gr.Group():
                with gr.Row():
                    scribble_en = gr.Checkbox(label="scribble enable", value=False)
                    scribble_img = gr.Image(label="scribble Image", type="pil")
                    scribble_sel = gr.Number(label="selector", value=None, interactive=False)
                    scribble_str = gr.Slider(0, 1, label="strength", value=0.5)
            close_btn = gr.Button("Close")

        controlnet_btn.click(lambda: Modal(visible=True), None, controlnet_modal)
        close_btn.click(lambda: Modal(visible=False), None, controlnet_modal)

        profile.change(
            _update_strength_defaults,
            inputs=[profile],
            outputs=[canny_str, depth_str, pose_str, scribble_str],
        )

        for cb in (canny_en, depth_en, pose_en, scribble_en):
            cb.change(
                _update_selectors,
                inputs=[canny_en, depth_en, pose_en, scribble_en],
                outputs=[canny_sel, depth_sel, pose_sel, scribble_sel],
            )

        def generate_image(
            profile, prompt, feeling, subject, environment, refine,
            num_images, seed, use_seed, spread, upscale_quality,
            canny_en, canny_img, canny_str,
            depth_en, depth_img, depth_str,
            pose_en, pose_img, pose_str,
            aspect_ratio, lora_strength,
            scribble_en, scribble_img,
            scribble_str, negative_prompt, style_preset
        ) -> tuple[list, str | None, list | None]:

            entries = [
                (GuidanceType.CANNY,    canny_en,    canny_img,    canny_str),
                (GuidanceType.DEPTH,    depth_en,    depth_img,    depth_str),
                (GuidanceType.POSE,     pose_en,     pose_img,     pose_str),
                (GuidanceType.SCRIBBLE, scribble_en, scribble_img, scribble_str),
            ]

            images_b64 = []
            controls_list = []
            for gtype, en, img, strength in entries:
                if en and img is not None:
                    images_b64.append(_pil_to_b64(img))
                    controls_list.append(
                        GuidanceSettings(selector=len(images_b64) - 1, type=gtype, strength=strength)
                    )

            controls = GuidanceInput(images=images_b64, controls=controls_list) if controls_list else None

            request = GenerateRequest(
                profile = profile,
                prompt = prompt,
                subject = subject,
                environment = environment,
                feeling = feeling,
                refine = refine,
                num_images = num_images,
                seed = seed if use_seed else None,
                spread = spread if use_seed else None,
                upscale_quality = upscale_quality if upscale_quality else None,
                aspect_ratio = aspect_ratio,
                negative_prompt = negative_prompt,
                lora_strength = lora_strength,
                controls = controls,
                style_preset = style_preset
            )

            response = requests.post(
                "http://localhost:5000/generate",
                json=request.model_dump(mode="json"),
            )

            if response.status_code != 200:
                return [], None, None

            data = response.json()
            gallery_images = []
            quality_data   = []

            for item in data["images"]:
                img_bytes = base64.b64decode(item["Image"])
                pil_img   = Image.open(io.BytesIO(img_bytes))
                caption   = f"seed: {item['seed']}" if item.get("seed") else ""
                gallery_images.append((pil_img, caption))

                if item.get("quality"):
                    quality_data.append({
                        "seed":  item.get("seed"),
                        "gates": item["quality"],
                    })

            refined = data.get("refined_prompt")
            return gallery_images, refined, quality_data if quality_data else None

        generate_button.click(
            generate_image,
            inputs=[
                profile, prompt, feeling, subject, environment, refine,
                num_images, seed, use_seed, spread, upscale_quality,
                canny_en, canny_img, canny_str,
                depth_en, depth_img, depth_str,
                pose_en, pose_img, pose_str,
                aspect_ratio, lora_strength,
                scribble_en, scribble_img, scribble_str, negative_prompt, style_preset,
            ],
            outputs=[gallery, refined_prompt_box, quality_output],
        )

    demo.launch(server_port=7860)