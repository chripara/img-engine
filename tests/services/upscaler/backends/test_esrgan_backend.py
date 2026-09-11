from PIL import Image

from app.services.upscaler.backends.esrgan_backend import ESRGANBackend
from app.services.upscaler.backends.tensor_upscale_runner import TensorUpscaleRunner
from app.services.upscaler.registries.upscaler_registry import _UPSCALERS
from utils.enums.upscale import Upscaler
from tests.services.upscaler.backends.rigs import _minimal_request


class _RecordingRunner(TensorUpscaleRunner):
    def __init__(self):
        self.load_calls: list = []
        self.run_calls: list = []
        self.unload_called = False

    def load(self, model_path):
        self.load_calls.append(model_path)

    def run(self, image):
        self.run_calls.append(image)
        return image

    def unload(self):
        self.unload_called = True


def test_load_passes_registered_model_path_for_upscaler_kind():
    runner = _RecordingRunner()
    backend = ESRGANBackend(upscaler=Upscaler.ANIME_ESRGAN, runner=runner)

    backend.load()

    assert runner.load_calls == [_UPSCALERS[Upscaler.ANIME_ESRGAN]]


def test_upscale_delegates_to_runner_and_returns_its_result():
    runner = _RecordingRunner()
    backend = ESRGANBackend(upscaler=Upscaler.ESRGAN, runner=runner)
    image = Image.new("RGB", (8, 8))

    result = backend.upscale(image, _minimal_request())

    assert runner.run_calls == [image]
    assert result is image


def test_upscale_writes_debug_png_with_esrgan_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = ESRGANBackend(upscaler=Upscaler.ESRGAN, runner=_RecordingRunner())

    backend.upscale(Image.new("RGB", (8, 8)), _minimal_request(), index=0, seed=42)

    assert (tmp_path / "output_images" / "seed_42_esrgan.png").exists()


def test_unload_delegates_to_runner():
    runner = _RecordingRunner()
    backend = ESRGANBackend(upscaler=Upscaler.ESRGAN, runner=runner)

    backend.unload()

    assert runner.unload_called is True
