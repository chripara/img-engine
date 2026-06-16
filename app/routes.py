from flask import Blueprint, jsonify
from app.services.img_service import generate

bp = Blueprint('routes', __name__)

@bp.route('/health', methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route('/generate', methods=["GET"])
def generate():
    return jsonify({"image": generate()})