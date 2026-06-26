from pydantic import BaseModel, Field
from utils.enums import Profile

class GenerateRequest(BaseModel):
    profile: Profile
    prompt: str = Field(..., max_length=600)
    subject: str | None
    environment: str | None
    feeling: str | None
    refine: bool = False
