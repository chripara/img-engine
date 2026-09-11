import torch
from typing import Type
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline, StableDiffusionXLControlNetPipeline
from compel import Compel, ReturnedEmbeddingsType

from app.services.image.backends.model_runner import ImageModelRunner


class SDXLModelRunner(ImageModelRunner):
    def __init__(self) -> None:
        self._pipe: DiffusionPipeline | None = None
        self._compel: Compel | None = None

    def load(
        self,
        checkpoint_id: str,
        vae_id: str | None,
        controlnet_model_cls: Type | None,
        controlnet_repo_ids: list[str],
        use_cpu_offload: bool,
    ) -> None:
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16) if vae_id else None

        if controlnet_repo_ids:
            controlnets = [
                controlnet_model_cls.from_pretrained(repo_id, torch_dtype=torch.float16)
                for repo_id in controlnet_repo_ids
            ]
            self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                checkpoint_id,
                controlnet=controlnets,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **({"vae": vae} if vae else {}),
            )
        else:
            self._pipe = DiffusionPipeline.from_pretrained(
                checkpoint_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **({"vae": vae} if vae else {}),
            )

        if use_cpu_offload:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to("cuda")

        self._compel = Compel(
            tokenizer=[self._pipe.tokenizer, self._pipe.tokenizer_2],
            text_encoder=[self._pipe.text_encoder, self._pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

    def apply_lora(self, lora_repo_id: str, adapter_name: str, strength: float) -> None:
        self._pipe.load_lora_weights(lora_repo_id, adapter_name=adapter_name)
        self._pipe.set_adapters(adapter_name, adapter_weights=strength)
        self._pipe.fuse_lora()

    def bind_scheduler(self, scheduler_cls: Type) -> None:
        self._pipe.scheduler = scheduler_cls.from_config(self._pipe.scheduler.config)

    def run(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int | None,
        control_images: list[Image.Image] | None,
        control_strengths: list[float] | None,
        index: int,
    ) -> Image.Image:
        conditioning, pooled = self._compel(prompt)
        negative_conditioning, negative_pooled = (
            self._compel(negative_prompt) if negative_prompt is not None else (None, None)
        )

        generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

        result = self._pipe(
            prompt_embeds=conditioning,
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_conditioning,
            negative_pooled_prompt_embeds=negative_pooled,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            **({"image": control_images} if control_images is not None else {}),
            **({"controlnet_conditioning_scale": control_strengths} if control_strengths is not None else {}),
            generator=generator,
        )

        return result.images[0]

    def unload(self) -> None:
        if self._pipe is not None:
            self._pipe.to("cpu")
        self._pipe = None
        self._compel = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
