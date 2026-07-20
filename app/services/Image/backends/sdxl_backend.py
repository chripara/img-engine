import io, torch, os, gc, warnings

from PIL import Image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from app.schemas.generate import GuidanceResult
from app.services.image.backends.base_backend import BaseBackend
from app.services.registries.image_registry import Dimensions, _SDXL_CONTROLNET_LIMIT
from app.services.registries.profile_registry import _PROFILES
from app.services.image.registries.checkpoint_registry import _CHECKPOINT
from utils.enums import Profile, GuidanceType
from compel import Compel, ReturnedEmbeddingsType

class SDXLBackend(BaseBackend):
    pipe: DiffusionPipeline

    def __init__ (self, profile: Profile):
        super().__init__()
        self._steps = _PROFILES[profile].steps
        self._cfg = _PROFILES[profile].cfg


    def load(self, profile: Profile, lora_weight: float | None, use_controlnet: bool, guidance_types: list[GuidanceType]) -> None:
        self._define_vae(profile)

        if use_controlnet:
            self._define_guidance(profile, guidance_types)

            self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                _CHECKPOINT[_PROFILES[profile].model],
                controlnet=self._controlnets,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **({"vae": self._vae} if self._vae else {}),
            )
        else:
            self._pipe = DiffusionPipeline.from_pretrained(
                _CHECKPOINT[_PROFILES[profile].model],
                torch_dtype=torch.float16,
                use_safetensors=True,
                **({"vae": self._vae} if self._vae else {}),
            )

        self._define_lora(profile, lora_weight)

        self._define_scheduler(profile)

        if len(guidance_types) < _SDXL_CONTROLNET_LIMIT:
            self._pipe.to("cuda")
        else:
            self._pipe.enable_model_cpu_offload()

        self.compel = Compel(
            tokenizer=[self._pipe.tokenizer, self._pipe.tokenizer_2],
            text_encoder=[self._pipe.text_encoder, self._pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

    def generate(self, prompt: str, negative_prompt: str | None, dimensions: Dimensions, seed: int | None, controls: list[GuidanceResult]) -> Image.Image:
        # Generate an image using the SDXL model
        conditioning, pooled = self.compel(prompt)
        negative_conditioning, negative_pooled = self.compel(negative_prompt) if negative_prompt is not None else None


        generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

        result = self._pipe(
            prompt_embeds = conditioning,
            pooled_prompt_embeds = pooled,
            negative_prompt_embeds = negative_conditioning,
            negative_pooled_prompt_embeds = negative_pooled,
            width = dimensions.width,
            height = dimensions.height,
            num_inference_steps = self._steps,
            guidance_scale = self._cfg,
            **({"image":[ctr.image for ctr in controls]} if controls is not None else {}),
            **({"controlnet_conditioning_scale": [ctr.strength for ctr in controls if ctr.strength is not None]} if controls is not None else {}),
            generator = generator,)

        image = result.images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", quality=95, dpi=(300, 300))
        png_bytes = buffer.getvalue()

        filename = f"seed_{seed if seed else 'NaN'}.png"
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)

        return image

    def unload(self) -> None:
        if self._pipe is not None:
            self._pipe.to("cpu")
        del self._pipe
        torch.cuda.empty_cache()
        gc.collect()