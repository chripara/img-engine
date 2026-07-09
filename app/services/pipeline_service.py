from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from app.schemas.generate import GenerateRequest, GuidanceResult
from app.services.image.img_service import generate_image
from utils.enums import UpscaleQuality, Profile
from utils.image_converter import ImageConverter
from app.services.upscaler.upscaler_service import upscale_image
from app.services.registries.profile_registry import _PROFILES
from app.services.registries.guidance_registry import  _GUIDANCE_DETAILS
from app.services.guidance.guidance_service import generate_guidance
import base64, random

class PipelineService():
    def generation_pipeline(req: GenerateRequest) -> list[str]:
        if req.seed is not None:
            seeds = [random.randint(req.seed - req.spread,
                req.seed + req.spread) if req.seed is not None and req.spread is not None else req.seed for _ in range(req.num_images)]
        else:
            seeds = [req.seed for _ in range(req.num_images)]

        with ThreadPoolExecutor() as executor:
            future_a = executor.submit(_refine_prompt,req)
            future_b = executor.submit(_preprocess, req)

            refined = future_a.result()
            req = req.model_copy(update={"prompt": refined})
            control_maps = future_b.result()
            _get_strength(req.profile,control_maps)

        images = generate_image(req, seeds,  control_maps)

        print(f"Images type: {type(images[0])}, count: {len(images)}")
        converter = ImageConverter.Pil_Image_to_Bytes_Png

        match req.upscale_quality:
            case UpscaleQuality.NONE:
                encoded = [base64.b64encode(converter(img)).decode() for img in images]
                print(f"Encoded images type: {type(encoded)}, count: {len(encoded)}")
                return encoded
            case UpscaleQuality.ENHANCED:
                imgs = upscale_image(req,_PROFILES[req.profile],images)
                encoded = [base64.b64encode(converter(img)).decode() for img in imgs]
                return encoded
            case UpscaleQuality.GENERATIVE:
                imgs = upscale_image(req, _PROFILES[req.profile], images)
                encoded = [base64.b64encode(converter(img)).decode() for img in imgs]
                return encoded

def _refine_prompt(req: GenerateRequest) -> str:
    refined_prompt = req.prompt
    if req.refine:
        from app.services.prompts.prompt_service import refine_prompt_with_ollama, refine_prompt_with_llama
        try:
            refined_prompt = refine_prompt_with_llama(req)
        except Exception as e_llama:
            print("llama broke ", e_llama)
            try:
                print("Trying now with fallback Ollama")
                refined_prompt = refine_prompt_with_ollama(req)
                print("Ollama Successful")
            except Exception as e_ollama:
                print("ollama broke ", e_ollama)
                print("No refinement engine is supported. passing propmpt as is!")
    return refined_prompt

def _preprocess(req: GenerateRequest) -> list[GuidanceResult]:
    return generate_guidance(req)

def _get_strength(profile: Profile, controls: list[GuidanceResult]) -> None:
    for guidance_result in controls:
        if guidance_result.strength is None:
            guidance_result.model_copy(update={"strength": _GUIDANCE_DETAILS[_PROFILES[profile].model].defaults[guidance_result.type]})

