from flask import Flask
import subprocess
import atexit
from config import DevelopmentConfig

_ollama = None

def start_ollama():
        global _ollama
        if _ollama is None:
                _ollama = subprocess.Popen([r"C:\Users\Chris\AppData\Local\Programs\Ollama\ollama.exe", "serve"])

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

        return app

                                
              