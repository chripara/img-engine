from unittest.mock import MagicMock
import pytest

from app.services.guidance.backends import sdxl_preprocessor_runner as runner_module
from app.services.guidance.backends.sdxl_preprocessor_runner import SDXLPreprocessorRunner
from utils.enums.guidance import GuidanceType


@pytest.fixture
def fake_preprocessors(monkeypatch):
    fake_detector = MagicMock(name="detector")
    fake_factory = MagicMock(return_value=fake_detector)
    monkeypatch.setitem(runner_module._SDXL_PREPROCESSORS, GuidanceType.CANNY, fake_factory)
    return fake_factory, fake_detector


def test_load_builds_detector_via_registry_factory(fake_preprocessors):
    fake_factory, fake_detector = fake_preprocessors
    runner = SDXLPreprocessorRunner()

    runner.load(GuidanceType.CANNY)

    fake_factory.assert_called_once()
    assert runner._detector is fake_detector


def test_run_calls_detector_with_image(fake_preprocessors):
    fake_factory, fake_detector = fake_preprocessors
    fake_detector.return_value = "preprocessed"
    runner = SDXLPreprocessorRunner()
    runner.load(GuidanceType.CANNY)

    result = runner.run("input_image")

    fake_detector.assert_called_once_with("input_image")
    assert result == "preprocessed"


def test_unload_clears_detector_and_clears_cuda_cache(monkeypatch, fake_preprocessors):
    torch_mock = MagicMock()
    monkeypatch.setattr(runner_module, "torch", torch_mock)
    runner = SDXLPreprocessorRunner()
    runner.load(GuidanceType.CANNY)

    runner.unload()

    assert runner._detector is None
    torch_mock.cuda.empty_cache.assert_called_once()
