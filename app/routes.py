from flask import Blueprint, jsonify, Response, request
from utils.enums import Checkpoint, Profile
from app.services.img_service import generate_image

bp = Blueprint('routes', __name__)

@bp.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route('/generate', methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    profile = data.get("profile")
    if not prompt or not profile:
        return Response(status=400)

    result = generate_image(str(prompt), Profile(profile))
    return  Response(result, mimetype="image/png")