from pydantic import BaseModel, Field
from utils.enums import Profile

class GenerateRequest(BaseModel):
    profile: Profile
    num_images: int = Field(..., ge=1, le=10)
    prompt: str = Field(..., max_length=600)
    subject: str | None
    environment: str | None
    feeling: str | None
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    refine: bool = False
