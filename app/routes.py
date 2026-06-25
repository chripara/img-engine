from flask import Blueprint, jsonify, Response, request
from utils.enums import Checkpoint, Profile
from app.services.Image.img_service import generate_image
from app.schemas.generate import GenerateRequest

bp = Blueprint('routes', __name__)

@bp.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route('/generate', methods=["POST"])
def generate():
    data = request.json
    req = GenerateRequest(**data)
    
    if not req.prompt or not req.profile:
        return Response(status=400)

    if req.refine:
        from app.services.prompts.prompt_service import refine as refine_prompt
        generate_request = req
        req.prompt = refine_prompt(generate_request)

    result = generate_image(req)
    return  Response(result, mimetype="image/png")