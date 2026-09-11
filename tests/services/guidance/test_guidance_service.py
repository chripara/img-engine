import base64
import io
import pytest
from PIL import Image

from app.schemas.generate import GenerateRequest, GuidanceInput, GuidanceSettings, GuidanceResult
from app.services.guidance import guidance_service
from utils.enums.guidance import GuidanceType
from utils.enums.profile import Profile


def _b64_png(size=(2, 2)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_request(**overrides) -> GenerateRequest:
    defaults = dict(profile=Profile.CHARACTER, num_images=1, prompt="a cat",
                     subject=None, environment=None, feeling=None)
    defaults.update(overrides)
    return GenerateRequest(**defaults)


class _FakeGuidanceEngine:
    instances: list["_FakeGuidanceEngine"] = []

    def __init__(self, req):
        self.req = req
        self.prepare_calls: list = []
        self.results_queue: list = []
        _FakeGuidanceEngine.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def prepare(self, control, image):
        self.prepare_calls.append((control, image))
        return self.results_queue.pop(0) if self.results_queue else None


@pytest.fixture(autouse=True)
def _reset_instances():
    _FakeGuidanceEngine.instances.clear()
    yield
    _FakeGuidanceEngine.instances.clear()


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(guidance_service, "GuidanceEngine", _FakeGuidanceEngine)


@pytest.fixture
def fake_engine_with_results(monkeypatch):
    def _install(results_queue):
        class _Engine(_FakeGuidanceEngine):
            def __init__(self, req):
                super().__init__(req)
                self.results_queue = list(results_queue)
        monkeypatch.setattr(guidance_service, "GuidanceEngine", _Engine)
    return _install


def test_returns_empty_list_when_no_controls_at_all(fake_engine):
    req = _make_request(controls=None)
    assert guidance_service.generate_guidance(req) == []
    assert _FakeGuidanceEngine.instances == []


def test_skips_control_with_selector_out_of_range(fake_engine):
    req = _make_request(controls=GuidanceInput(
        images=[_b64_png()],
        controls=[GuidanceSettings(selector=5, type=GuidanceType.CANNY, strength=None)],
    ))

    result = guidance_service.generate_guidance(req)

    assert result == []
    assert _FakeGuidanceEngine.instances[0].prepare_calls == []


def test_prepares_each_valid_control_and_collects_non_none_results(fake_engine_with_results):
    fake_result = GuidanceResult(image=Image.new("RGB", (2, 2)), type=GuidanceType.CANNY, strength=0.4)
    fake_engine_with_results([fake_result, None])

    req = _make_request(controls=GuidanceInput(
        images=[_b64_png(), _b64_png()],
        controls=[
            GuidanceSettings(selector=0, type=GuidanceType.CANNY, strength=0.4),
            GuidanceSettings(selector=1, type=GuidanceType.DEPTH, strength=None),
        ],
    ))

    result = guidance_service.generate_guidance(req)

    assert result == [fake_result]
    assert len(_FakeGuidanceEngine.instances[0].prepare_calls) == 2


def test_passes_the_image_selected_by_control_selector_not_the_first_one(fake_engine_with_results):
    fake_engine_with_results([None])
    img_a = _b64_png((2, 2))
    img_b = _b64_png((4, 4))
    req = _make_request(controls=GuidanceInput(
        images=[img_a, img_b],
        controls=[GuidanceSettings(selector=1, type=GuidanceType.CANNY, strength=None)],
    ))

    guidance_service.generate_guidance(req)

    _, passed_image = _FakeGuidanceEngine.instances[0].prepare_calls[0]
    assert passed_image.size == (4, 4)
