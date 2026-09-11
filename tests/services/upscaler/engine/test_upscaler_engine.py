import pytest
from PIL import Image

from app.schemas.generate import GenerateRequest
from app.services.registries.profile_registry import _PROFILES
from app.services.upscaler.engine import upscaler_engine as upscaler_engine_module
from app.services.upscaler.engine.upscaler_engine import UpscalerEngine
from utils.enums.profile import Profile
from utils.enums.upscale import UpscaleQuality


class _FakeUpscaleBackend:
    instances: list["_FakeUpscaleBackend"] = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.loaded = False
        self.unloaded = False
        self.upscale_calls: list[dict] = []
        _FakeUpscaleBackend.instances.append(self)

    def load(self):
        self.loaded = True

    def unload(self):
        self.unloaded = True

    def upscale(self, img, req, index, seed):
        self.upscale_calls.append(dict(img=img, req=req, index=index, seed=seed))
        return img


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakeUpscaleBackend.instances.clear()
    yield
    _FakeUpscaleBackend.instances.clear()


@pytest.fixture
def fake_backends(monkeypatch):
    monkeypatch.setattr(upscaler_engine_module, "ESRGANBackend", _FakeUpscaleBackend)
    monkeypatch.setattr(upscaler_engine_module, "LatentDiffusionBackend", _FakeUpscaleBackend)


def _make_request(upscale_quality) -> GenerateRequest:
    return GenerateRequest(profile=Profile.CHARACTER, num_images=1, prompt="a cat",
                            subject=None, environment=None, feeling=None,
                            upscale_quality=upscale_quality)


def test_none_quality_skips_backend_entirely(fake_backends):
    engine = UpscalerEngine(_make_request(UpscaleQuality.NONE), _PROFILES[Profile.CHARACTER])

    assert engine._upscaler is None
    assert _FakeUpscaleBackend.instances == []


def test_none_quality_upscale_image_returns_input_unchanged(fake_backends):
    engine = UpscalerEngine(_make_request(UpscaleQuality.NONE), _PROFILES[Profile.CHARACTER])
    img = Image.new("RGB", (4, 4))

    assert engine.upscale_image(img, _make_request(UpscaleQuality.NONE)) is img


def test_enhanced_quality_constructs_backend_with_profile_esrgan_upscaler(fake_backends):
    spec = _PROFILES[Profile.CHARACTER]
    engine = UpscalerEngine(_make_request(UpscaleQuality.ENHANCED), spec)

    assert isinstance(engine._upscaler, _FakeUpscaleBackend)
    assert engine._upscaler.init_kwargs["upscaler"] == spec.esrgan_upscaler


def test_generative_quality_constructs_backend_with_denoising_strength(fake_backends):
    engine = UpscalerEngine(_make_request(UpscaleQuality.GENERATIVE), _PROFILES[Profile.CHARACTER])

    assert isinstance(engine._upscaler, _FakeUpscaleBackend)
    assert "denoising_strength" in engine._upscaler.init_kwargs


def test_context_manager_loads_when_backend_present(fake_backends):
    with UpscalerEngine(_make_request(UpscaleQuality.ENHANCED), _PROFILES[Profile.CHARACTER]) as engine:
        assert engine._upscaler.loaded is True
        assert engine._upscaler.unloaded is False


def test_context_manager_is_safe_when_backend_is_none(fake_backends):
    """Το ρητό `if self._upscaler is not None:` guard στο __exit__ υπήρχε
    ήδη στον κώδικα — αυτό το test το κλειδώνει."""
    with UpscalerEngine(_make_request(UpscaleQuality.NONE), _PROFILES[Profile.CHARACTER]):
        pass


def test_upscale_image_delegates_to_backend_when_present(fake_backends):
    img = Image.new("RGB", (4, 4))
    req = _make_request(UpscaleQuality.ENHANCED)
    with UpscalerEngine(req, _PROFILES[Profile.CHARACTER]) as engine:
        engine.upscale_image(img, req, index=1, seed=5)

    backend = _FakeUpscaleBackend.instances[0]
    assert len(backend.upscale_calls) == 1
    assert backend.upscale_calls[0]["img"] is img
    assert backend.upscale_calls[0]["index"] == 1
    assert backend.upscale_calls[0]["seed"] == 5
