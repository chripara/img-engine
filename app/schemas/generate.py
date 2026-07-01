from pydantic import BaseModel, Field
from utils.enums import Profile, UpscaleQuality

class GenerateRequest(BaseModel):
    profile: Profile
    num_images: int = Field(..., ge=1, le=10)
    prompt: str = Field(..., max_length=600)
    subject: str | None
    environment: str | None
    feeling: str | None
    refine: bool = False
    seed: int | None = Field(default=None, le=2**32 - 1)
    spread: int | None = Field(default=None, ge=0)
    upscale_quality: UpscaleQuality | None = Field(default=UpscaleQuality.NONE)
