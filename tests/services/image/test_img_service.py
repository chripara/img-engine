import pytest
from PIL import Image

from app.schemas.generate import GenerateRequest, GuidanceResult
from app.services.image import img_service
from utils.enums.guidance import GuidanceType
from utils.enums.profile import Profile


def _make_request(**overrides) -> GenerateRequest:
    defaults = dict(profile=Profile.CHARACTER, num_images=2, prompt="a cat",
                     subject=None, environment=None, feeling=None)
    defaults.update(overrides)
    return GenerateRequest(**defaults)


class _FakeImageEngine:
    instances: list["_FakeImageEngine"] = []

    def __init__(self, req, guidance_types):
        self.req = req
        self.guidance_types = guidance_types
        self.generate_calls: list[dict] = []
        _FakeImageEngine.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def generate_image(self, req, seed, controls, index):
        self.generate_calls.append(dict(seed=seed, controls=controls, index=index))
        return Image.new("RGB", (8, 8))


@pytest.fixture(autouse=True)
def _reset_fake_engine_instances():
    _FakeImageEngine.instances.clear()
    yield
    _FakeImageEngine.instances.clear()


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(img_service, "ImageEngine", _FakeImageEngine)


def test_generate_image_builds_guidance_types_from_controls(fake_engine):
    controls = [
        GuidanceResult(image=Image.new("RGB", (2, 2)), type=GuidanceType.CANNY, strength=0.5),
        GuidanceResult(image=Image.new("RGB", (2, 2)), type=GuidanceType.DEPTH, strength=None),
    ]

    img_service.generate_image(_make_request(), seeds=[1, 2], controls=controls)

    assert _FakeImageEngine.instances[0].guidance_types == [GuidanceType.CANNY, GuidanceType.DEPTH]


def test_generate_image_returns_one_image_per_seed(fake_engine):
    result = img_service.generate_image(_make_request(), seeds=[10, 20, 30], controls=[])

    assert len(result) == 3
    engine = _FakeImageEngine.instances[0]
    assert [c["seed"] for c in engine.generate_calls] == [10, 20, 30]
    assert [c["index"] for c in engine.generate_calls] == [0, 1, 2]


def test_generate_image_crashes_when_controls_is_none(fake_engine):
     img_service.generate_image(_make_request(), seeds=[1], controls=None)

     assert _FakeImageEngine.instances[0].guidance_types == []
