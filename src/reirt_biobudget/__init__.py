"""reirt-biobudget: 3D Deformable Biophysical Dose-Accumulation & Remaining-Budget Engine.

Research software for multi-course radiotherapy re-irradiation analysis, DIR-uncertainty
envelopes, multi-regime radiobiology (LQ, LQL, Lea-Catcheside, variable proton RBE),
LKB/gEUD cumulative NTCP, Saltelli-Sobol global sensitivity, clinical DICOM-RT
interoperability (REG/RTDOSE/RTSTRUCT), Missing-DICOM / Legacy-PDF 3D prior-dose envelope
reconstruction, and frontier agent orchestration via MCP.

NOT A MEDICAL DEVICE. For academic research and educational physics verification only.
"""

from reirt_biobudget.ir import (
    AbstentionError,
    AuditVerdict,
    CourseSpec,
    DIRQualityMetrics,
    LegacyCourseRecord,
    LegacyDVHConstraint,
    LegacyOARReconstructionSummary,
    LegacyReconstructionAudit,
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
    SourceSpanEvidence,
    Volume3D,
)

__version__ = "0.3.0"

__all__ = [
    "AbstentionError",
    "AuditVerdict",
    "CourseSpec",
    "DIRQualityMetrics",
    "LKBModelSpec",
    "LegacyCourseRecord",
    "LegacyDVHConstraint",
    "LegacyOARReconstructionSummary",
    "LegacyReconstructionAudit",
    "Modality",
    "OARBudgetSummary",
    "OARConstraint",
    "OARNTCPSummary",
    "PriorCourseInput",
    "ProvenanceRecord",
    "RadiobiologyModel",
    "ReIrradiationResult",
    "SobolSensitivityResult",
    "SourceSpanEvidence",
    "Volume3D",
    "__version__",
]
