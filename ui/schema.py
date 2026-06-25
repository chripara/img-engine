from pydantic import BaseModel

class GenerateRequest(BaseModel):
    profile: str
    prompt: str
    subject: str | None
    environment: str | None
    feeling: str | None
    refine: bool = False