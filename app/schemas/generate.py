from PIL import Image
from pydantic import BaseModel, Field, ConfigDict
from utils.enums import GuidanceType

from utils.enums import Profile, UpscaleQuality

class GuidanceSettings(BaseModel):
    selector: int = Field(default=0)
    type: GuidanceType
    strength: float | None = None

class GuidanceInput(BaseModel):
    images: list[str]
    controls: list[GuidanceSettings]
    strength: float | None = None

class GuidanceResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    images: list[Image.Image]
    controls: list[GuidanceType]

class GenerateRequest(BaseModel):
    profile: Profile
    num_images: int = Field(..., ge=1, le=10)
    prompt: str = Field(..., max_length=600)
    subject: str | None
    environment: str | None
    feeling: str | None
    refine: bool = False
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    spread: int | None = Field(default=None, ge=0)
    controls: GuidanceInput | None = Field(default=None)
    upscale_quality: UpscaleQuality | None = Field(default=UpscaleQuality.NONE)

