import numpy as np
import pytest
from PIL import Image

from app.services.validation.validators.tiling_validator import tiling_validator
from app.services.validation.registries.validator_registry import resolve_gate_status
from utils.enums.gate import GateType, GateStatus


def _random_noise_image(size=(128, 128), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def _tiled_pattern_image(size=(128, 128), tile=8) -> Image.Image:
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, size=(tile, tile), dtype=np.uint8)
    reps_y = size[1] // tile + 1
    reps_x = size[0] // tile + 1
    arr = np.tile(base, (reps_y, reps_x))[: size[1], : size[0]]
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_returns_gate_result_for_tiling_with_consistent_status_and_score():
    result = tiling_validator(_random_noise_image())
    assert result.gate == GateType.TILING
    expected_status = resolve_gate_status(GateType.TILING, result.score)
    assert result.passed == (expected_status == GateStatus.PASS)


def test_tiled_repeating_pattern_scores_lower_than_random_noise():
    noise_score = tiling_validator(_random_noise_image()).score
    tiled_score = tiling_validator(_tiled_pattern_image()).score
    assert tiled_score < noise_score


@pytest.mark.parametrize("size", [(64, 64), (100, 60), (33, 129)])
def test_handles_non_square_and_non_power_of_two_sizes_without_raising(size):
    tiling_validator(_random_noise_image(size))
