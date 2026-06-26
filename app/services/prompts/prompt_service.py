from flask import Flask, request, jsonify
import requests
from app.schemas.generate import GenerateRequest

app = Flask(__name__)

SYSTEM_PROMPT = """You are an expert image prompt engineer for Stable Diffusion XL.
Given a profile, subject, feeling and environment, generate a single detailed image prompt.
Rules:
- Output ONLY the prompt string, no explanations, no quotes, no labels
- Incorporate the feeling as visual style/mood/palette descriptors
- Incorporate the environment as background/context
- Keep the subject as the focal point
- Keep the output up to 600 characters
- If the prompt contains a named character, deity or any other known figure, preserve their iconic attributes 
(weapons, appearance, powers, person) in the output prompt.
- Use comma-separated descriptive keywords"""

def refine(generate_request: GenerateRequest) -> str:
    
    user_message = f"""Profile: {generate_request.profile}
    Subject: {generate_request.subject}
    Feeling: {generate_request.feeling}
    Environment: {generate_request.environment}"""

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "system": SYSTEM_PROMPT,
        "prompt": user_message,
        "stream": False
    })

    return response.json()["response"].strip()

if __name__ == "__main__":
    app.run(port=5001)