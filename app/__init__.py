from flask import Flask
import subprocess
import atexit
from config import DevelopmentConfig

# Ollama is a local LLM server that we will start when the Flask app starts and stop when the Flask app stops.
# it is necessary up until i create the Prompt Refinement Engine that has each own pipeline...
# So it is just a temporary solution to get the app running with the current prompt refinement engine.
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

                                
              