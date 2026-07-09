from app.schemas.generate import GenerateRequest, GuidanceInput, GuidanceResult
from app.services.guidance.engine.guidance_engine import GuidanceEngine
import base64, io
from PIL import Image
def generate_guidance(req: GenerateRequest) -> list[GuidanceResult]:

    if(req.controls):
        with GuidanceEngine(req) as engine:
            control_maps = []
            for control in req.controls.controls:
                if control.type is None:
                    continue
                if not (0 <= control.selector < len(req.controls.images)):
                    continue

                raw = base64.b64decode(req.controls.images[control.selector])
                image = Image.open(io.BytesIO(raw))

                result = engine.prepare(control, image)
                if result is not None:
                    control_maps.append(result)

        return control_maps
    else:
        return []