from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.services.upscaler.backends.sd_upscale_pipeline_runner import SDUpscalePipelineRunner

MODULE = "app.services.upscaler.backends.sd_upscale_pipeline_runner"


def test_run_tile_before_load_raises_runtime_error():
    runner = SDUpscalePipelineRunner()
    with pytest.raises(RuntimeError):
        runner.run_tile("a cat", Image.new("RGB", (2, 2)), noise_level=20, steps=8)


def test_load_builds_pipeline_and_enables_cpu_offload():
    with patch(f"{MODULE}.StableDiffusionUpscalePipeline") as pipe_cls, patch(f"{MODULE}.torch") as torch_mod:
        fake_pipe = MagicMock(name="pipe")
        pipe_cls.from_pretrained.return_value = fake_pipe

        runner = SDUpscalePipelineRunner()
        runner.load("some/model-path")

        pipe_cls.from_pretrained.assert_called_once_with("some/model-path", torch_dtype=torch_mod.float16)
        fake_pipe.enable_model_cpu_offload.assert_called_once()


def test_run_tile_passes_prompt_tile_noise_level_and_steps():
    with patch(f"{MODULE}.StableDiffusionUpscalePipeline") as pipe_cls, patch(f"{MODULE}.torch"):
        fake_pipe = MagicMock(name="pipe")
        expected_image = Image.new("RGB", (8, 8))
        fake_pipe.return_value.images = [expected_image]
        pipe_cls.from_pretrained.return_value = fake_pipe

        runner = SDUpscalePipelineRunner()
        runner.load("some/model-path")
        tile = Image.new("RGB", (4, 4))

        result = runner.run_tile("a cat", tile, noise_level=30, steps=8)

        _, kwargs = fake_pipe.call_args
        assert kwargs["prompt"] == "a cat"
        assert kwargs["image"] is tile
        assert kwargs["noise_level"] == 30
        assert kwargs["num_inference_steps"] == 8
        assert result is expected_image


def test_unload_moves_pipe_to_cpu_and_clears_cuda_cache():
    with patch(f"{MODULE}.StableDiffusionUpscalePipeline") as pipe_cls, patch(f"{MODULE}.torch") as torch_mod:
        fake_pipe = MagicMock(name="pipe")
        pipe_cls.from_pretrained.return_value = fake_pipe

        runner = SDUpscalePipelineRunner()
        runner.load("some/model-path")
        runner.unload()

        fake_pipe.to.assert_called_once_with("cpu")
        assert runner._pipe is None
        torch_mod.cuda.empty_cache.assert_called_once()


def test_unload_is_safe_before_load():
    runner = SDUpscalePipelineRunner()
    runner.unload()
