import pytest
from PIL import Image

from app.schemas.generate import GenerateRequest, GuidanceSettings
from app.services.guidance.engine import guidance_engine as guidance_engine_module
from app.services.guidance.engine.guidance_engine import GuidanceEngine
from utils.enums.guidance import GuidanceType
from utils.enums.profile import Profile


class _FakeGuidanceBackend():
    instances: list["_FakeGuidanceBackend"] = []

    def __init__(self):
        self.load_calls: list = []
        self.preprocess_calls: list = []
        self.unload_call_count = 0
        self.next_preprocess_result = None
        _FakeGuidanceBackend.instances.append(self)

    def load(self, guidance_type):
        self.load_calls.append(guidance_type)

    def unload(self):
        self.unload_call_count += 1

    def preprocess(self, image):
        self.preprocess_calls.append(image)
        return self.next_preprocess_result


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakeGuidanceBackend.instances.clear()
    yield
    _FakeGuidanceBackend.instances.clear()


def _make_request(**overrides) -> GenerateRequest:
    defaults = dict(profile=Profile.CHARACTER, num_images=1, prompt="a cat",
                     subject=None, environment=None, feeling=None)
    defaults.update(overrides)
    return GenerateRequest(**defaults)


@pytest.fixture
def fake_backend_registered(monkeypatch):
    checkpoint = guidance_engine_module._PROFILES[Profile.CHARACTER].model
    monkeypatch.setitem(guidance_engine_module._GUIDANCE_BACKEND, checkpoint, _FakeGuidanceBackend)


def test_constructor_resolves_registered_backend_class(fake_backend_registered):
    GuidanceEngine(_make_request())
    assert len(_FakeGuidanceBackend.instances) == 1


def test_exit_calls_unload(fake_backend_registered):
    with GuidanceEngine(_make_request()) as engine:
        backend = engine._backend
        assert backend.unload_call_count == 0
    assert backend.unload_call_count == 1

def test_prepare_loads_with_controls_type_and_unloads_twice(fake_backend_registered):
    with GuidanceEngine(_make_request()) as engine:
        backend = engine._backend
        backend.next_preprocess_result = Image.new("RGB", (2, 2))
        control = GuidanceSettings(selector=0, type=GuidanceType.CANNY, strength=0.6)
        engine.prepare(control, Image.new("RGB", (2, 2)))

    assert backend.load_calls == [GuidanceType.CANNY]
    assert backend.unload_call_count == 2

def test_prepare_returns_none_when_preprocess_returns_none(fake_backend_registered):
    with GuidanceEngine(_make_request()) as engine:
        engine._backend.next_preprocess_result = None
        control = GuidanceSettings(selector=0, type=GuidanceType.CANNY, strength=0.6)
        result = engine.prepare(control, Image.new("RGB", (2, 2)))

    assert result is None


def test_prepare_builds_guidance_result_with_control_type_and_strength(fake_backend_registered):
    processed = Image.new("RGB", (2, 2))
    with GuidanceEngine(_make_request()) as engine:
        engine._backend.next_preprocess_result = processed
        control = GuidanceSettings(selector=0, type=GuidanceType.DEPTH, strength=0.9)
        result = engine.prepare(control, Image.new("RGB", (2, 2)))

    assert result.type == GuidanceType.DEPTH
    assert result.strength == 0.9
    assert result.image is processed
