from flask import Flask
import subprocess
import atexit, os, shutil
from config import DevelopmentConfig

_ollama = None


OLLAMA_PATH = os.getenv("OLLAMA_PATH") or shutil.which("ollama") or "ollama"

def start_ollama():
    global _ollama
    if _ollama is None:
        try:
            _ollama = subprocess.Popen([OLLAMA_PATH, "serve"])
        except (FileNotFoundError, OSError) as e:
            print(f"Ollama not available ({e}) — PRE will fall back to Groq only.")
            _ollama = None

def stop_ollama():
        if _ollama is not None:
                _ollama.terminate()

def create_app():
        app = Flask(__name__)
        app.config.from_object(DevelopmentConfig)
          
        start_ollama()
        atexit.register(stop_ollama)

        from app import routes
        app.register_blueprint(routes.bp)
        routes.api.register(app)

        return app

                                
              