from dotenv import load_dotenv
from flask import Flask
import requests, os
from app.schemas.generate import GenerateRequest
from groq import Groq

app = Flask(__name__)

SYSTEM_PROMPT = """You are an SDXL prompt engineer. Your output is ONLY a comma-separated tag list — no sentences, no narrative, no explanations, no quotes.

Rules:
- Output format: tag, tag, tag, tag, ...
- Max 150 tokens
- Named subjects (gods, heroes, characters): preserve their exact name and iconic attributes (weapons, armor, powers, appearance)
- Feeling → visual style, color palette, lighting mood
- Environment → background, atmosphere, setting details
- Subject is always the focal point
- Use visual descriptors: lighting quality, color palette, texture, composition, art style
- Examples: golden hour lighting, dramatic rim light, volumetric fog, cinematic composition, 8k detailed, oil painting style
- Avoid generic words like "epic", "heroic", "intense" — describe HOW it looks visually
- No filler words, no punctuation except commas"""

def refine_prompt_with_llama(generate_request: GenerateRequest) -> str | None:
    load_dotenv()
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _generate_message(generate_request)},
        ],
        temperature=0.1,
        seed=generate_request.seed,
        max_tokens=150,
    )
    answer = chat_completion.choices[0].message.content
    return answer


def refine_prompt_with_ollama(generate_request: GenerateRequest) -> str:

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "system": SYSTEM_PROMPT,
        "prompt": _generate_message(generate_request),
        "stream": False,
        "options": {
            "seed": generate_request.seed,
            "temperature" : 0
        }
    })

    return response.json()["response"].strip()

def _generate_message(generate_request: GenerateRequest) -> str:
    return f"""Generate a comma-separated SDXL tag list only. No sentences. No story.    
        Subject: {generate_request.subject}
        Feeling: {generate_request.feeling}
        Environment: {generate_request.environment}
        Prompt: {generate_request.prompt}    
        Output only tags separated by commas:"""

if __name__ == "__main__":
    app.run(port=5001)