from pydantic import BaseModel, Field

from utils.enums import GuidanceType


class GuidanceSettings(BaseModel):
    selector: int = Field(default=0)
    type: GuidanceType
    strength: float | None = None

class GuidanceInput(BaseModel):
    images: list[str]
    controls: list[GuidanceSettings]

class GenerateRequest(BaseModel):
    profile: str
    num_images: int = Field(..., ge=1, le=10)
    prompt: str = Field(..., max_length=600)
    subject: str | None
    environment: str | None
    feeling: str | None
    seed: int | None = Field(default=None, ge=0, le=2 ** 32 - 1)
    spread: int | None = Field(0, ge=0)
    upscale_quality: str | None = Field(default="none")
    refine: bool = False
    controls: GuidanceInput | None = None

