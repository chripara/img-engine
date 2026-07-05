from app.schemas.generate import GenerateRequest, GuidanceInput
from app.services.guidance.engine.guidance_engine import GuidanceEngine


def generate_guidance(req: GenerateRequest) -> list[Image.Image]:
    with GuidanceEngine(req) as engine:

        control_maps = []
        for control in req.controls.controls:
            if control.type is None:
                continue
            if not (0 <= control.selector < len(req.controls.images)):
                continue
            image = req.controls.images[control.selector]
            result = engine.prepare_guidance(control, image)
            control_maps.append(result)

    return control_maps