from PIL import Image

from app.services.guidance.backends.preprocessor_runner import GuidancePreprocessorRunner
from app.services.guidance.backends.sdxl_guidance_backend import SDXLGuidanceBackend
from utils.enums.guidance import GuidanceType


class _RecordingRunner(GuidancePreprocessorRunner):
    def __init__(self):
        self.load_calls: list = []
        self.run_calls: list = []
        self.unload_called = False

    def load(self, guidance_type):
        self.load_calls.append(guidance_type)

    def run(self, image):
        self.run_calls.append(image)
        return image

    def unload(self):
        self.unload_called = True


def test_preprocess_before_load_returns_none_and_never_touches_runner():
    runner = _RecordingRunner()
    backend = SDXLGuidanceBackend(runner=runner)

    result = backend.preprocess(Image.new("RGB", (2, 2)))

    assert result is None
    assert runner.run_calls == []


def test_load_delegates_guidance_type_to_runner():
    runner = _RecordingRunner()
    backend = SDXLGuidanceBackend(runner=runner)

    backend.load(GuidanceType.POSE)

    assert runner.load_calls == [GuidanceType.POSE]


def test_preprocess_after_load_delegates_to_runner_and_returns_its_result():
    runner = _RecordingRunner()
    backend = SDXLGuidanceBackend(runner=runner)
    backend.load(GuidanceType.CANNY)
    image = Image.new("RGB", (2, 2))

    result = backend.preprocess(image)

    assert runner.run_calls == [image]
    assert result is image


def test_unload_delegates_to_runner_and_resets_loaded_flag():
    runner = _RecordingRunner()
    backend = SDXLGuidanceBackend(runner=runner)
    backend.load(GuidanceType.CANNY)

    backend.unload()

    assert runner.unload_called is True
    assert backend.preprocess(Image.new("RGB", (2, 2))) is None
