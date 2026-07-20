from PIL import Image
from pydantic import BaseModel, Field, ConfigDict, field_validator
from utils.enums import GuidanceType, AspectRatio, GateType

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
    image: Image.Image
    type: GuidanceType
    strength: float | None

class GenerateRequest(BaseModel):
    profile: Profile
    num_images: int = Field(..., ge=1, le=10)
    prompt: str = Field(..., max_length=600)
    negative_prompt: str | None = Field(default=None, max_length=600)
    subject: str | None
    environment: str | None
    feeling: str | None
    refine: bool = False
    seed: int | None = Field(default=None, le=2**32 - 1)
    spread: int | None = Field(default=None, ge=0)
    controls: GuidanceInput | None = Field(default=None)
    aspect_ratio: AspectRatio | None = Field(default = AspectRatio.SQUARE.value)
    lora_strength: float | None = Field(default = None, ge = 0, le = 1.0)
    upscale_quality: UpscaleQuality | None = Field(default=UpscaleQuality.NONE)

    @field_validator("aspect_ratio", mode="before")
    @classmethod
    def validate_aspect_ratio(cls, v):
        try:
            return AspectRatio(v)
        except ValueError:
            return AspectRatio.SQUARE

class GateResult(BaseModel):
    gate: GateType | None
    score: float | None = None
    passed: bool | None = None
    suggested: str | None = None

class ImageResult(BaseModel):
    image: str
    seed: int | None = None
    quality: list[GateResult] | None = None

class GenerateResult(BaseModel):
    images: list[ImageResult]
    refined_prompt: str | None = None
