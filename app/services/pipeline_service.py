from app.schemas.generate import GenerateRequest
from app.services.image.img_service import generate_image
from utils.enums import UpscaleQuality
from utils.image_converter import ImageConverter
from app.services.upscaler.upscaler_service import upscale_image
from app.services.registries.profile_registry import _PROFILES
import base64

class PipelineService():
    def generation_pipeline(req: GenerateRequest) -> list[str]:
        if req.refine:
            from app.services.prompts.prompt_service import refine as refine_prompt
            refined_prompt = refine_prompt(req)
            req = req.model_copy(update={"prompt": refined_prompt})

        images = generate_image(req)

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

