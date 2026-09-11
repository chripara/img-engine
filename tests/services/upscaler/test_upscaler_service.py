import pytest
from PIL import Image

from app.schemas.generate import GenerateRequest
from app.services.registries.profile_registry import _PROFILES
from app.services.upscaler import upscaler_service
from utils.enums.profile import Profile


def _make_request(**overrides) -> GenerateRequest:
    defaults = dict(profile=Profile.CHARACTER, num_images=1, prompt="a cat",
                     subject=None, environment=None, feeling=None)
    defaults.update(overrides)
    return GenerateRequest(**defaults)


class _FakeUpscalerEngine:
    instances: list["_FakeUpscalerEngine"] = []

    def __init__(self, req, spec):
        self.req = req
        self.spec = spec
        self.upscale_calls: list[dict] = []
        _FakeUpscalerEngine.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def upscale_image(self, img, req, index, seed):
        self.upscale_calls.append(dict(img=img, index=index, seed=seed))
        return img


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakeUpscalerEngine.instances.clear()
    yield
    _FakeUpscalerEngine.instances.clear()


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(upscaler_service, "UpscalerEngine", _FakeUpscalerEngine)


def test_upscale_image_calls_engine_once_per_image_with_matching_seed_and_index(fake_engine):
    imgs = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]
    seeds = [10, 20, 30]

    result = upscaler_service.upscale_image(_make_request(), _PROFILES[Profile.CHARACTER], imgs, seeds)

    assert len(result) == 3
    engine = _FakeUpscalerEngine.instances[0]
    assert [c["seed"] for c in engine.upscale_calls] == seeds
    assert [c["index"] for c in engine.upscale_calls] == [0, 1, 2]


def test_upscale_image_passes_spec_through_to_engine_constructor(fake_engine):
    spec = _PROFILES[Profile.PRODUCT]

    upscaler_service.upscale_image(_make_request(), spec, [], [])

    assert _FakeUpscalerEngine.instances[0].spec is spec
