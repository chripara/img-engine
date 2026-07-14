import base64

from flask import Blueprint, jsonify, Response, request
from pydantic import ValidationError
from app.services.pipeline_service import PipelineService
from app.schemas.generate import GenerateRequest
from flask_pydantic_spec import FlaskPydanticSpec, Request, Response

bp = Blueprint('routes', __name__)

api = FlaskPydanticSpec("flask", title="img-engine API", version="1.0", path="docs")


@bp.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route('/generate', methods=["POST"])
@api.validate(body=Request(GenerateRequest), resp=Response(HTTP_200=None))
def generate():
    try:
        req = GenerateRequest(**request.json)
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 422
    
    if not req.prompt or not req.profile:
        return Response(status=400)

    response = PipelineService.generation_pipeline(req)

    return jsonify(response.model_dump())