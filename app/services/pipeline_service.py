from concurrent.futures import ThreadPoolExecutor
from app.schemas.generate import GenerateRequest, GuidanceResult, GenerateResult, ImageResult
from app.services.image.img_service import generate_image
from utils.enums.upscale import UpscaleQuality
from utils.enums.profile import Profile
from utils.image_converter import ImageConverter
from app.services.upscaler.upscaler_service import upscale_image
from app.services.registries.profile_registry import _PROFILES
from app.services.registries.guidance_registry import  _GUIDANCE_DETAILS
from app.services.guidance.guidance_service import generate_guidance
from app.services.validation.validator import validate
import base64, random

class PipelineService():
    def generation_pipeline(req: GenerateRequest) -> GenerateResult:
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

        converter = ImageConverter.Pil_Image_to_Bytes_Png

        match req.upscale_quality:
            case UpscaleQuality.ENHANCED | UpscaleQuality.GENERATIVE:
                images = upscale_image(req, _PROFILES[req.profile], images, seeds)

        image_results: list[ImageResult] = []
        for i in range(len(images)):
            validations = validate(images[i])
            encoded = base64.b64encode(converter(images[i])).decode()
            image_results.append(ImageResult(
                image=encoded,
                seed=seeds[i],
                quality=validations
            ))
        return GenerateResult(images = image_results, refined_prompt = refined)

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
    for i, guidance_result in enumerate(controls):
        if guidance_result.strength is None:
            controls[i] = guidance_result.model_copy(
                update={"strength": _GUIDANCE_DETAILS[_PROFILES[profile].model].defaults[guidance_result.type]}
            )
