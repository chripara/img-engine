from flask import Blueprint, jsonify, Response, request
from utils.enums import Checkpoint
from app.services.img_service import generate_image

bp = Blueprint('routes', __name__)

@bp.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route('/generate', methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt")
    checkpoint = data.get("checkpoint")
    if not prompt or not checkpoint:
        return Response(status=400)

    result = generate_image(str(prompt), Checkpoint(checkpoint))
    return  Response(result, mimetype="image/png")