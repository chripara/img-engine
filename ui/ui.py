import gradio as gr
import requests, io
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
                generate_button = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(label="Output Image")

        def generate_image(prompt, profile):
           print(prompt, profile)
           response = requests.post("http://localhost:5000/generate", json={"prompt": prompt, "profile": profile})
           return Image.open(io.BytesIO(response.content)) if response.status_code == 200 else None

        
        generate_button.click(generate_image, inputs=[prompt, profile], outputs=output_image)

    demo.launch(server_port=7860)