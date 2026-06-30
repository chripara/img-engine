import io, torch, hashlib, time, os, gc
from PIL import Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from app.services.image.backends.base_backend import BaseBackend
from app.services.registries.profile_registry import _PROFILES
from app.services.image.registries.checkpoint_registry import _CHECKPOINT
from utils.enums import Profile
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from compel import Compel, ReturnedEmbeddingsType

class SDXLBackend(BaseBackend):
    pipe: DiffusionPipeline

    def __init__ (self, profile: Profile):
        super().__init__()              

        vae = AutoencoderKL.from_pretrained(
            _PROFILES[profile].vae_id,
            torch_dtype = torch.float16
        ) if _PROFILES[profile].vae_id else None

        self.pipe = DiffusionPipeline.from_pretrained(
            _CHECKPOINT[_PROFILES[profile].model],
            torch_dtype = torch.float16,            
            use_safetensors = True,     # χρήση safetensors            
            **({"vae": vae} if vae else {}),
        )
        self.steps = _PROFILES[profile].steps
        self.cfg = _PROFILES[profile].cfg
        self.pipe.scheduler = _PROFILES[profile].scheduler.from_config(self.pipe.scheduler.config) 
        self.pipe.to("cuda")
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

    def generate(self, prompt: str, seed: int | None) -> Image.Image:

        # Generate an image using the SDXL model
        conditioning, pooled = self.compel(prompt)
        print(type(self.pipe))
        print(hasattr(self.pipe, 'tokenizer_2'))
        generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None
    
        result = self.pipe(
            prompt_embeds = conditioning,
            pooled_prompt_embeds = pooled, 
            num_inference_steps = self.steps, 
            guidance_scale = self.cfg,
            generator = generator,)

        image = result.images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95, dpi=(300, 300))
        png_bytes = buffer.getvalue()

        hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        filename = f"{int(time.time())}_{hash}.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)

        return image

    def unload(self) -> None:
        self.pipe = None
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()