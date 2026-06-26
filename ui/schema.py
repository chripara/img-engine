from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    profile: str
    num_images: int = Field(..., ge=1, le=10)
    prompt: str = Field(..., max_length=600)
    subject: str | None
    environment: str | None
    feeling: str | None
    refine: bool = False