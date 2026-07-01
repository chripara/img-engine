import base64

from flask import Blueprint, jsonify, Response, request
from pydantic import ValidationError
from app.services.pipeline_service import PipelineService
from app.schemas.generate import GenerateRequest

bp = Blueprint('routes', __name__)

@bp.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route('/generate', methods=["POST"])
def generate():
    try:
        req = GenerateRequest(**request.json)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 422
    
    if not req.prompt or not req.profile:
        return Response(status=400)

    if req.refine:
        from app.services.prompts.prompt_service import refine as refine_prompt
        generate_request = req
        req.prompt = refine_prompt(generate_request)

    images = PipelineService.generation_pipeline(req)
    print(f"Encoded[0] length: {len(images[0])}, preview: {images[0][:20]}")

    return jsonify({"images": images})