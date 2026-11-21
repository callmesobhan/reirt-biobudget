"""reirt-biobudget: 3D Deformable Biophysical Dose-Accumulation & Remaining-Budget Engine.

Research software for multi-course radiotherapy re-irradiation analysis, DIR-uncertainty
envelopes, multi-regime radiobiology (LQ, LQL, Lea-Catcheside, variable proton RBE),
LKB/gEUD cumulative NTCP, Saltelli-Sobol global sensitivity, clinical DICOM-RT
interoperability (REG/RTDOSE/RTSTRUCT), and frontier agent orchestration via MCP.

NOT A MEDICAL DEVICE. For academic research and educational physics verification only.
"""

from reirt_biobudget.ir import (
    AbstentionError,
    AuditVerdict,
    CourseSpec,
    DIRQualityMetrics,
    LKBModelSpec,
    Modality,
    OARBudgetSummary,
    OARConstraint,
    OARNTCPSummary,
    PriorCourseInput,
    ProvenanceRecord,
    RadiobiologyModel,
    ReIrradiationResult,
    SobolSensitivityResult,
    Volume3D,
)

__version__ = "0.2.0"

__all__ = [
    "AbstentionError",
    "AuditVerdict",
    "CourseSpec",
    "DIRQualityMetrics",
    "LKBModelSpec",
    "Modality",
    "OARBudgetSummary",
    "OARConstraint",
    "OARNTCPSummary",
    "PriorCourseInput",
    "ProvenanceRecord",
    "RadiobiologyModel",
    "ReIrradiationResult",
    "SobolSensitivityResult",
    "Volume3D",
    "__version__",
]
