from pydantic import BaseModel
from utils.enums.gate import GateType, GateStatus

class GateStatusMessage(BaseModel):
    status: GateStatus
    message: str | None

_GATE_MESSAGES: dict[GateType, dict[GateStatus, str | None]] = {
    GateType.TILING: {
        GateStatus.PASS: None,
        GateStatus.WARNING: "Tiling pattern detected",
        GateStatus.FAIL: "Severe tiling artifact — image unusable",
    },
    GateType.HANDS: {
        GateStatus.PASS: None,
        GateStatus.WARNING: "Hand anatomy issues detected",
        GateStatus.FAIL: "Severe hand deformation",
        GateStatus.NOT_APPLICABLE: "No hands detected in image",
    },
    GateType.FACE: {
        GateStatus.PASS: None,
        GateStatus.WARNING: "Face quality degraded",
        GateStatus.FAIL: "Face unrecognizable",
    },
    GateType.CLIP: {
        GateStatus.PASS: None,
        GateStatus.WARNING: "Prompt alignment weak",
        GateStatus.FAIL: "image does not match prompt",
    },
    GateType.IQA: {
        GateStatus.PASS: None,
        GateStatus.WARNING: "Low perceptual quality",
        GateStatus.FAIL: "image quality unacceptable",
    },
}

_GATE_THRESHOLDS: dict[GateType, dict[GateStatus, float]]= {
    GateType.TILING: {
        GateStatus.WARNING: 0.1,    # pass ≥ 0.75
        GateStatus.FAIL:  0.05
    },
    GateType.HANDS: {
        GateStatus.WARNING: 0.65,    # pass ≥ 0.65
        GateStatus.FAIL:  0.35
    },
    GateType.FACE: {
        GateStatus.WARNING: 0.70,    # pass ≥ 0.70
        GateStatus.FAIL:  0.40
    },
    GateType.CLIP: {
        GateStatus.WARNING: 0.25,    # pass ≥ 0.25
        GateStatus.FAIL:  0.20
    },
    GateType.IQA: {
        GateStatus.WARNING: 0.60,    # pass ≥ 0.60
        GateStatus.FAIL:  0.40
    },
}