import base64
from random import random
from unittest import result

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
                    label="Profile",
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
                upscale_quality = gr.Dropdown(
                    label="Upscale Quality",
                    choices=[
                        ("none"),
                        ("enhanced"),
                        ("generative"),
                    ]
                )
                num_images = gr.Dropdown(
                    label="Number of Images",
                    choices=[
                        ("1", 1),
                        ("2", 2),
                        ("3", 3),
                        ("4", 4), 
                        ("5", 5),
                        ("6", 6),
                        ("7", 7),
                        ("8", 8),
                        ("9", 9),
                        ("10", 10),
                    ]
                )
                with gr.Row(equal_height=True):
                        use_seed = gr.Checkbox(label="Use Seed", value=False)
                        seed = gr.Number(label="Seed", value=42, precision=0)
                        spread = gr.Number(label="Spread", value=0, precision=0)
                generate_button = gr.Button("Generate")
            with gr.Column():
                gallery = gr.Gallery(label="Output Images")


        def generate_image(profile: str, prompt: str, feeling: str, subject: str, environment: str, refine: bool, num_images: int, seed: int, use_seed: bool, spread: int, upscale_quality: str) -> list[Image.Image]:
            request = GenerateRequest(
                profile=profile,    
                prompt=prompt,
                subject=subject,
                environment=environment,
                feeling=feeling,
                refine=refine,
                num_images=num_images,
                seed=seed if use_seed else None,
                spread=spread if use_seed else None,
                upscale_quality = upscale_quality if upscale_quality else None
                )
            
            print(request)
            
            response = requests.post("http://localhost:5000/generate", json=request.model_dump())
            images = []

            if response.status_code == 200:
                data = response.json()
                result = [base64.b64decode(img) for img in data["images"]]

                for content in result:
                    images.append(Image.open(io.BytesIO(content)))
            return images

        generate_button.click(generate_image, inputs=[profile, prompt, feeling, subject, environment, refine, num_images, seed , use_seed, spread, upscale_quality], outputs=[gallery])

    demo.launch(server_port=7860)