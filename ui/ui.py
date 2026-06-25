import gradio as gr
import requests, io
from ui.schema import GenerateRequest
from utils.enums import Checkpoint, Profile
from PIL import Image


def launch_ui():

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                profile = gr.Dropdown(
                    label="profile",
                    choices=[
                        ("Character / Hero", Profile.CHARACTER.value),
                        ("Product / Item", Profile.PRODUCT.value),
                        ("Scene / Card Frame", Profile.SCENE_FRAME.value),
                    ]
                )
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
                generate_button = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(label="Output Image")

        def generate_image(profile: str, prompt: str, feeling: str, subject: str, environment: str, refine: bool):
           request = GenerateRequest(
                profile=profile,    
                prompt=prompt,
                subject=subject,
                environment=environment,
                feeling=feeling,
                refine=refine
                )
           print(request)
           response = requests.post("http://localhost:5000/generate", json=request.model_dump())
           return Image.open(io.BytesIO(response.content)) if response.status_code == 200 else None

        generate_button.click(generate_image, inputs=[profile, prompt, feeling, subject, environment, refine], outputs=output_image)

    demo.launch(server_port=7860)