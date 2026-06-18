import gradio as gr
import requests, io
from utils.enums import Checkpoint
from PIL import Image


def launch_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                checkpoint = gr.Dropdown(
                    label="checkpoint",
                    choices=[
                        ("General", Checkpoint.SDXL_BASE.value),
                        ("Character / Hero", Checkpoint.ALBEDO_BASE.value),
                        ("Environment / Background", Checkpoint.DREAMSHAPER_XL.value),
                        ("Key Art / Marketing", Checkpoint.JUGGERNAUT_XL.value)
                    ]
                )
                generate_button = gr.Button("Generate")
            with gr.Column():
                output_image = gr.Image(label="Output Image")

        def generate_image(prompt, checkpoint):
           print(prompt, checkpoint)
           response = requests.post("http://localhost:5000/generate", json={"prompt": prompt, "checkpoint": checkpoint})
           return Image.open(io.BytesIO(response.content)) if response.status_code == 200 else None

        
        generate_button.click(generate_image, inputs=[prompt, checkpoint], outputs=output_image)

    demo.launch(server_port=7860)