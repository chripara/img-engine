from flask import Flask
import subprocess
import atexit, os, shutil
from config import Config, DevelopmentConfig

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

_ollama = None

OLLAMA_PATH = os.getenv("OLLAMA_PATH") or shutil.which("ollama") or "ollama"

def start_ollama():
    global _ollama
    if _ollama is None:
        try:
            _ollama = subprocess.Popen([OLLAMA_PATH, "serve"])
        except (FileNotFoundError, OSError) as e:
            logger.warning("Ollama not available (%s) — PRE will rely on Groq only, with no local fallback.", e)
            _ollama = None

def stop_ollama():
        if _ollama is not None:
                _ollama.terminate()

def create_app():
        app = Flask(__name__)
        # Debug mode must be explicitly opted into via APP_ENV=development — safe (DEBUG=False) by default.
        config_class = DevelopmentConfig if os.getenv("APP_ENV") == "development" else Config
        app.config.from_object(config_class)

        start_ollama()
        atexit.register(stop_ollama)

        from app import routes
        app.register_blueprint(routes.bp)
        routes.api.register(app)

        return app
