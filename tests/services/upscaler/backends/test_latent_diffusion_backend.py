from unittest.mock import MagicMock
import pytest
from PIL import Image

from app.services.upscaler.backends.latent_diffusion_backend import LatentDiffusionBackend
from app.services.upscaler.backends.tiled_diffusion_upscale_runner import TiledDiffusionUpscaleRunner
from tests.services.upscaler.backends.rigs import _minimal_request


class _IdentityTileRunner(TiledDiffusionUpscaleRunner):

    def __init__(self):
        self.run_tile_calls: list[dict] = []

    def load(self, model_path):
        pass

    def run_tile(self, prompt, tile, noise_level, steps):
        self.run_tile_calls.append(dict(prompt=prompt, tile=tile, noise_level=noise_level, steps=steps))
        w, h = tile.size
        return tile.resize((w * 4, h * 4))

    def unload(self):
        pass


def test_upscale_returns_image_scaled_exactly_4x():
    runner = _IdentityTileRunner()
    backend = LatentDiffusionBackend(runner=runner)
    src = Image.new("RGB", (100, 60))

    result = backend.upscale(src, _minimal_request())

    assert result.size == (400, 240)


def test_upscale_passes_prompt_and_noise_level_derived_from_denoising_strength():
    runner = _IdentityTileRunner()
    backend = LatentDiffusionBackend(denoising_strength=0.3, runner=runner)
    req = _minimal_request()

    backend.upscale(Image.new("RGB", (64, 64)), req)

    assert all(c["noise_level"] == 30 for c in runner.run_tile_calls)
    assert all(c["steps"] == 8 for c in runner.run_tile_calls)
    assert all(c["prompt"] == req.prompt for c in runner.run_tile_calls)


@pytest.mark.parametrize("size", [(64, 64), (100, 60), (37, 89), (513, 513)])
def test_upscale_covers_every_tile_for_non_power_of_two_dimensions(size):
    w, h = size
    tile_size_x, tile_size_y = w // 2, h // 2
    overlap_x, overlap_y = w // 16, h // 16
    expected_tiles = len(range(0, h, tile_size_y - overlap_y)) * len(range(0, w, tile_size_x - overlap_x))

    runner = _IdentityTileRunner()
    backend = LatentDiffusionBackend(runner=runner)
    backend.upscale(Image.new("RGB", size), _minimal_request())

    assert len(runner.run_tile_calls) == expected_tiles


def test_upscale_writes_debug_png_with_latent_suffix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    backend = LatentDiffusionBackend(runner=_IdentityTileRunner())

    backend.upscale(Image.new("RGB", (32, 32)), _minimal_request(), index=0, seed=7)

    assert (tmp_path / "output_images" / "seed_7_latent.png").exists()


def test_unload_delegates_to_runner():
    runner = MagicMock(spec=TiledDiffusionUpscaleRunner)
    backend = LatentDiffusionBackend(runner=runner)

    backend.unload()

    runner.unload.assert_called_once()
