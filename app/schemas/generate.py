from pydantic import BaseModel
from utils.enums import Profile

class GenerateRequest(BaseModel):
    profile: Profile
    prompt: str
    subject: str | None
    environment: str | None
    feeling: str | None
    refine: bool = False
