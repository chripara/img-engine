from email.mime import image
import profile
from click import prompt
from app.services.backends.base_backend import BaseBackend
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from app.services.backends.profile_registry import _PROFILES
from utils.enums import Checkpoint, ModelSource, Profile
from app.services.backends.checkpoint_registry import _CHECKPOINT
from PIL import Image
import io, torch, hashlib, time, os
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

class SDXLBackend(BaseBackend):
    pipe: DiffusionPipeline

    def __init__ (self, profile: Profile):
        super().__init__()              

        vae = AutoencoderKL.from_pretrained(
            _PROFILES[profile].vae_id,
            torch_dtype = torch.float16
        ) if _PROFILES[profile].vae_id else None

        self.pipe = DiffusionPipeline.from_pretrained(
            #pretrained_model_name_or_path =
            _CHECKPOINT[_PROFILES[profile].model],
            torch_dtype = torch.float16,
            #**({"variant": _PROFILES[profile].variant} if _PROFILES[profile].variant else {}),
            use_safetensors = True,     # χρήση safetensors            
            **({"vae": vae} if vae else {}),
            # device_map="balanced",        # auto device placement,
        )
        #self.pipe.enable_attention_slicing() #For large models, this can help reduce memory usage
        self.steps = _PROFILES[profile].steps
        self.cfg = _PROFILES[profile].cfg
        self.pipe.scheduler = _PROFILES[profile].scheduler.from_config(self.pipe.scheduler.config) 
        self.pipe.to("cuda")

    def generate(self, prompt: str) -> bytes:

        # Generate a dump image 
        # img = Image.new("RGBA", (200, 200), (0, 128, 255, 255))  # μπλε τετράγωνο
        
        # Generate an image using the SDXL model
        result = self.pipe(prompt=prompt, num_inference_steps=self.steps, guidance_scale=self.cfg)
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