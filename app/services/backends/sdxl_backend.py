from email.mime import image
from click import prompt
from app.services.backends.base_backend import BaseBackend
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from utils.enums import Checkpoint
from app.services.backends.checkpoint_registry import _CHECKPOINT
from PIL import Image
import io, torch, hashlib, time, os

class SDXLBackend(BaseBackend):
    pipe: DiffusionPipeline

    def __init__ (self, checkpoint: Checkpoint):
        super().__init__()
        self.pipe = DiffusionPipeline.from_pretrained(
            _CHECKPOINT[checkpoint],
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.pipe.enable_attention_slicing()
        self.pipe.to("cuda")

    def generate(self, prompt: str) -> bytes:

        # Generate a dump image 
        # img = Image.new("RGBA", (200, 200), (0, 128, 255, 255))  # μπλε τετράγωνο
        
        # Generate an image using the SDXL model
        result = self.pipe(prompt=prompt)
        image = result.images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"{int(time.time())}_{hash}.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)

        return png_bytes