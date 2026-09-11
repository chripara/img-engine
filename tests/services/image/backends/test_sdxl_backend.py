from unittest.mock import MagicMock
import pytest
from PIL import Image

from app.schemas.generate import GuidanceResult
from app.services.image.backends.model_runner import ImageModelRunner
from app.services.image.backends.sdxl_backend import SDXLBackend
from app.services.image.registries.guidance_registry import _GUIDANCE_MODELS, _SDXL_CONTROLNET_MODELS
from app.services.image.registries.stype_presets import _STYLE_PRESET_REGISTRY
from app.services.registries.image_registry import Dimensions, _SDXL_CONTROLNET_LIMIT
from app.services.registries.profile_registry import _PROFILES
from utils.enums.guidance import GuidanceType
from utils.enums.profile import Profile
from utils.enums.style_presets import StylePreset


class _RecordingRunner(ImageModelRunner):

    def __init__(self):
        self.load_calls: list[dict] = []
        self.apply_lora_calls: list[tuple] = []
        self.bind_scheduler_calls: list = []
        self.run_calls: list[dict] = []
        self.unload_called = False

    def load(self, **kwargs):
        self.load_calls.append(kwargs)

    def apply_lora(self, lora_repo_id, adapter_name, strength):
        self.apply_lora_calls.append((lora_repo_id, adapter_name, strength))

    def bind_scheduler(self, scheduler_cls):
        self.bind_scheduler_calls.append(scheduler_cls)

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        return Image.new("RGB", (8, 8))

    def unload(self):
        self.unload_called = True


def _backend() -> tuple[SDXLBackend, _RecordingRunner]:
    runner = _RecordingRunner()
    return SDXLBackend(profile=Profile.CHARACTER, runner=runner), runner


@pytest.mark.parametrize(
    "guidance_types, expected_use_cpu_offload",
    [
        ([], False),
        ([GuidanceType.CANNY], False),
        ([GuidanceType.CANNY, GuidanceType.DEPTH], False),
        ([GuidanceType.CANNY, GuidanceType.DEPTH, GuidanceType.POSE], True),  
        ([GuidanceType.CANNY, GuidanceType.DEPTH, GuidanceType.POSE, GuidanceType.SCRIBBLE], True),
    ],
)
def test_load_use_cpu_offload_boundary(guidance_types, expected_use_cpu_offload):
    backend, runner = _backend()
    backend.load(
        profile=Profile.CHARACTER, style_preset=None, lora_weight=None,
        use_controlnet=True, guidance_types=guidance_types,
    )
    assert runner.load_calls[0]["use_cpu_offload"] is expected_use_cpu_offload


def test_load_without_controlnet_passes_empty_repo_ids_and_no_model_cls():
    backend, runner = _backend()
    backend.load(
        profile=Profile.CHARACTER, style_preset=None, lora_weight=None,
        use_controlnet=False, guidance_types=[GuidanceType.CANNY],
    )
    assert runner.load_calls[0]["controlnet_repo_ids"] == []
    assert runner.load_calls[0]["controlnet_model_cls"] is None


def test_load_with_controlnet_resolves_model_cls_and_repo_ids_in_order():
    backend, runner = _backend()
    backend.load(
        profile=Profile.CHARACTER, style_preset=None, lora_weight=None,
        use_controlnet=True, guidance_types=[GuidanceType.CANNY, GuidanceType.DEPTH],
    )
    call = runner.load_calls[0]
    assert call["controlnet_model_cls"] is _GUIDANCE_MODELS[Profile.CHARACTER]
    assert call["controlnet_repo_ids"] == [
        _SDXL_CONTROLNET_MODELS[GuidanceType.CANNY],
        _SDXL_CONTROLNET_MODELS[GuidanceType.DEPTH],
    ]

def test_load_skips_lora_when_no_style_preset():
    backend, runner = _backend()
    backend.load(profile=Profile.CHARACTER, style_preset=None, lora_weight=None, use_controlnet=False, guidance_types=[])
    assert runner.apply_lora_calls == []


def test_load_applies_lora_with_default_strength_when_none_given():
    backend, runner = _backend()
    backend.load(
        profile=Profile.CHARACTER, style_preset=StylePreset.FANTASY, lora_weight=None,
        use_controlnet=False, guidance_types=[],
    )
    assert runner.apply_lora_calls == [
        (_STYLE_PRESET_REGISTRY[StylePreset.FANTASY], StylePreset.FANTASY.value, 0.8)
    ]


def test_load_applies_lora_with_given_strength():
    backend, runner = _backend()
    backend.load(
        profile=Profile.CHARACTER, style_preset=StylePreset.CYBERPUNK, lora_weight=0.4,
        use_controlnet=False, guidance_types=[],
    )
    assert runner.apply_lora_calls[0][2] == 0.4


def test_load_swallows_lora_exception_and_still_binds_scheduler():
    backend, runner = _backend()
    runner.apply_lora = MagicMock(side_effect=RuntimeError("boom"))

    backend.load(
        profile=Profile.CHARACTER, style_preset=StylePreset.FANTASY, lora_weight=None,
        use_controlnet=False, guidance_types=[],
    )

    assert runner.bind_scheduler_calls == [_PROFILES[Profile.CHARACTER].scheduler]


def test_load_binds_scheduler_from_profile():
    backend, runner = _backend()
    backend.load(profile=Profile.CHARACTER, style_preset=None, lora_weight=None, use_controlnet=False, guidance_types=[])
    assert runner.bind_scheduler_calls == [_PROFILES[Profile.CHARACTER].scheduler]


def test_generate_extracts_control_images_and_only_non_none_strengths():
    backend, runner = _backend()
    img1, img2 = Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))
    controls = [
        GuidanceResult(image=img1, type=GuidanceType.CANNY, strength=0.7),
        GuidanceResult(image=img2, type=GuidanceType.DEPTH, strength=None),
    ]

    backend.generate("a cat", None, Dimensions(width=64, height=64), seed=1, controls=controls)

    call = runner.run_calls[0]
    assert call["control_images"] == [img1, img2]
    assert call["control_strengths"] == [0.7]


def test_generate_with_none_controls_passes_none_through():
    backend, runner = _backend()
    backend.generate("a cat", None, Dimensions(width=64, height=64), seed=1, controls=None)

    call = runner.run_calls[0]
    assert call["control_images"] is None
    assert call["control_strengths"] is None


def test_generate_writes_debug_png_to_output_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend, _ = _backend()

    backend.generate("a cat", None, Dimensions(width=64, height=64), seed=999, controls=None)

    assert (tmp_path / "output_images" / "seed_999.png").exists()


def test_unload_delegates_to_runner():
    backend, runner = _backend()
    backend.unload()
    assert runner.unload_called is True
