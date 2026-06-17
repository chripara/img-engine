from flask import Blueprint, jsonify, Response, request
from app.services.img_service import generate_image

bp = Blueprint('routes', __name__)

@bp.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route('/generate', methods=["GET"])
def generate():
    prompt = request.args.get("prompt");
    if(prompt == None):
        return Response(status=400)

    result = generate_image(str(prompt))
    return  Response(result, mimetype="image/png")