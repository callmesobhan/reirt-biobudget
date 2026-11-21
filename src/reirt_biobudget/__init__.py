"""reirt-biobudget: 3D Deformable Biophysical Dose-Accumulation & Remaining-Budget Engine.

Research software for multi-course radiotherapy re-irradiation analysis, DIR-uncertainty
envelopes, multi-regime radiobiology (LQ, LQL, Lea-Catcheside, variable proton RBE),
and frontier agent orchestration via MCP.

NOT A MEDICAL DEVICE. For academic research and educational physics verification only.
"""

from reirt_biobudget.ir import (
    AbstentionError,
    AuditVerdict,
    CourseSpec,
    DIRQualityMetrics,
    Modality,
    OARBudgetSummary,
    OARConstraint,
    ProvenanceRecord,
    RadiobiologyModel,
    ReIrradiationResult,
    Volume3D,
)

__version__ = "0.1.0"

__all__ = [
    "AbstentionError",
    "AuditVerdict",
    "CourseSpec",
    "DIRQualityMetrics",
    "Modality",
    "OARBudgetSummary",
    "OARConstraint",
    "ProvenanceRecord",
    "RadiobiologyModel",
    "ReIrradiationResult",
    "Volume3D",
    "__version__",
]
