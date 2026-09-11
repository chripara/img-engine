import pytest
from PIL import Image

from app.schemas.generate import GenerateRequest
from app.services.image.backends.base_backend import BaseBackend
from app.services.image.engine import image_engine as image_engine_module
from app.services.image.engine.image_engine import ImageEngine
from app.services.registries.image_registry import _ASPECT_RATIOS
from utils.enums.aspect_ratio import AspectRatio
from utils.enums.profile import Profile


class _FakeBackend():
    def __init__(self, profile):
        self.profile = profile
        self.load_calls: list[dict] = []
        self.unloaded = False
        self.last_generate_kwargs: dict | None = None

    def load(self, profile, style_preset, lora_weight, use_controlnet, guidance_types) -> None:
        self.load_calls.append(dict(profile=profile, style_preset=style_preset, lora_weight=lora_weight,
                                     use_controlnet=use_controlnet, guidance_types=guidance_types))

    def unload(self) -> None:
        self.unloaded = True

    def generate(self, prompt, negative_prompt, dimensions, seed, controls, index=0) -> Image.Image:
        self.last_generate_kwargs = dict(prompt=prompt, negative_prompt=negative_prompt, dimensions=dimensions,
                                          seed=seed, controls=controls, index=index)
        return Image.new("RGB", (dimensions.width, dimensions.height))


def _make_request(**overrides) -> GenerateRequest:
    defaults = dict(profile=Profile.CHARACTER, num_images=1, prompt="a cat",
                     subject=None, environment=None, feeling=None)
    defaults.update(overrides)
    return GenerateRequest(**defaults)


@pytest.fixture
def fake_backend_registered(monkeypatch):
    checkpoint = image_engine_module._PROFILES[Profile.CHARACTER].model
    original_entry = image_engine_module._BACKENDS[checkpoint]
    monkeypatch.setitem(image_engine_module._BACKENDS, checkpoint, {**original_entry, "backend": _FakeBackend})
    return checkpoint


def test_get_backend_resolves_to_registered_class(fake_backend_registered):
    engine = ImageEngine(_make_request(), guidance_types=[])
    assert isinstance(engine._backend, _FakeBackend)


def test_context_manager_calls_load_then_unload(fake_backend_registered):
    with ImageEngine(_make_request(), guidance_types=[]) as engine:
        assert len(engine._backend.load_calls) == 1
        assert engine._backend.unloaded is False
    assert engine._backend.unloaded is True


def test_generate_image_raises_when_prompt_is_empty(fake_backend_registered):
    req = _make_request(prompt="")
    with ImageEngine(req, guidance_types=[]) as engine:
        with pytest.raises(ValueError):
            engine.generate_image(req)


@pytest.mark.parametrize(
    "aspect_ratio",
    [AspectRatio.SQUARE, AspectRatio.LANDSCAPE, AspectRatio.PORTRAIT],
)
def test_generate_image_resolves_dimensions_from_aspect_ratio(fake_backend_registered, aspect_ratio):
    req = _make_request(aspect_ratio=aspect_ratio.value)
    with ImageEngine(req, guidance_types=[]) as engine:
        engine.generate_image(req, seed=42)
        assert engine._backend.last_generate_kwargs["dimensions"] == _ASPECT_RATIOS[aspect_ratio]


def test_generate_image_passes_seed_controls_and_index_through(fake_backend_registered):
    req = _make_request()
    with ImageEngine(req, guidance_types=[]) as engine:
        engine.generate_image(req, seed=7, controls=None, index=2)
        kwargs = engine._backend.last_generate_kwargs
        assert kwargs["seed"] == 7
        assert kwargs["index"] == 2
