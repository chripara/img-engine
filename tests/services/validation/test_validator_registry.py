import pytest

from app.services.validation.registries.validator_registry import _GATE_THRESHOLDS, resolve_gate_status
from utils.enums.gate import GateType, GateStatus


@pytest.mark.parametrize("gate", list(GateType))
def test_score_at_or_above_warning_threshold_passes(gate):
    warning = _GATE_THRESHOLDS[gate][GateStatus.WARNING]
    assert resolve_gate_status(gate, warning) == GateStatus.PASS
    assert resolve_gate_status(gate, warning + 0.001) == GateStatus.PASS


@pytest.mark.parametrize("gate", list(GateType))
def test_score_between_fail_and_warning_is_warning(gate):
    fail = _GATE_THRESHOLDS[gate][GateStatus.FAIL]
    warning = _GATE_THRESHOLDS[gate][GateStatus.WARNING]
    midpoint = (fail + warning) / 2
    assert resolve_gate_status(gate, midpoint) == GateStatus.WARNING


@pytest.mark.parametrize("gate", list(GateType))
def test_score_below_fail_threshold_is_fail(gate):
    fail = _GATE_THRESHOLDS[gate][GateStatus.FAIL]
    assert resolve_gate_status(gate, fail - 0.001) == GateStatus.FAIL


@pytest.mark.parametrize("gate", list(GateType))
def test_score_exactly_at_fail_threshold_is_warning_not_fail(gate):
    fail = _GATE_THRESHOLDS[gate][GateStatus.FAIL]
    assert resolve_gate_status(gate, fail) == GateStatus.WARNING
