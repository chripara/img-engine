from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.services.upscaler.backends.spandrel_tensor_runner import SpandrelTensorRunner

MODULE = "app.services.upscaler.backends.spandrel_tensor_runner"


def test_run_before_load_raises_runtime_error():
    runner = SpandrelTensorRunner()
    with pytest.raises(RuntimeError):
        runner.run(Image.new("RGB", (2, 2)))


def test_load_builds_model_from_file_and_moves_to_cuda():
    with patch(f"{MODULE}.ModelLoader") as loader_cls:
        fake_model = MagicMock(name="model")
        loader_cls.return_value.load_from_file.return_value.cuda.return_value = fake_model

        runner = SpandrelTensorRunner()
        runner.load("some/path.pth")

        loader_cls.return_value.load_from_file.assert_called_once_with("some/path.pth")
        assert runner._model is fake_model


def test_run_converts_image_through_model_and_returns_pil_image():
    with patch(f"{MODULE}.ModelLoader") as loader_cls, \
         patch(f"{MODULE}.torch"), \
         patch(f"{MODULE}.Image") as image_mod:
        fake_model = MagicMock(name="model")
        loader_cls.return_value.load_from_file.return_value.cuda.return_value = fake_model
        image_mod.fromarray.return_value = "converted_image"

        runner = SpandrelTensorRunner()
        runner.load("some/path.pth")
        result = runner.run(Image.new("RGB", (2, 2)))

        fake_model.assert_called_once()
        assert result == "converted_image"


def test_unload_moves_model_to_cpu_and_clears_cuda_cache():
    with patch(f"{MODULE}.ModelLoader") as loader_cls, patch(f"{MODULE}.torch") as torch_mod:
        fake_model = MagicMock(name="model")
        loader_cls.return_value.load_from_file.return_value.cuda.return_value = fake_model

        runner = SpandrelTensorRunner()
        runner.load("some/path.pth")
        runner.unload()

        fake_model.cpu.assert_called_once()
        assert runner._model is None
        torch_mod.cuda.empty_cache.assert_called_once()


def test_unload_is_safe_before_load():
    runner = SpandrelTensorRunner()
    runner.unload()
