"""Canonical Physics Intermediate Representation (IR) for reirt-biobudget.

All volumetric grids, courses, constraints, DIR quality diagnostics, provenance
ledgers, and audit verdicts pass through strictly typed Pydantic models and
validated NumPy containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class Modality(str, Enum):
    """Supported radiotherapy delivery modalities."""

    PHOTON = "photon"
    PROTON_FIXED_RBE = "proton_fixed_rbe"
    PROTON_VARIABLE_RBE = "proton_variable_rbe"
    BRACHY_HDR = "brachy_hdr"
    BRACHY_LDR = "brachy_ldr"


class RadiobiologyModel(str, Enum):
    """Supported voxel-wise radiobiological equivalence kernels."""

    LQ = "lq"
    LQL = "lql"
    LEA_CATCHESIDE = "lea_catcheside"
    MCNAMARA_PROTON = "mcnamara_proton"


class ReIrradiationType(str, Enum):
    """ESTRO/EORTC 2022 re-irradiation classification (Andratschke et al., Lancet Oncol 2022)."""

    TYPE_1_OVERLAP = "Type 1: New course with geometrical overlap of irradiated volumes"
    TYPE_2_TOXICITY_CONCERN = (
        "Type 2: No geometrical overlap, but concern of toxicity from cumulative doses"
    )
    REPEAT_ORGAN = "Repeat organ irradiation (no overlap, negligible cumulative OAR overlap)"


class AuditVerdict(str, Enum):
    """Deterministic safety & feasibility verdict per OAR or global dossier."""

    PASS = "PASS"
    CAUTION = "CAUTION"
    VIOLATION = "VIOLATION"
    ABSTAIN = "ABSTAIN"


class AbstentionError(RuntimeError):
    """Raised when hard physical, spatial, or clinical provenance invariants fail.

    Instead of extrapolating silently, the engine and frontier agent harness must
    abstain and report the exact blocking invariant code.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(f"[ABSTAIN:{code}] {message}")


@dataclass
class Volume3D:
    """Validated 3D scalar medical physics volume in canonical (z, y, x) order.

    Axis convention:
        z (axis 0): Superior -> Inferior (SI) or Inferior -> Superior consistently tagged
        y (axis 1): Anterior -> Posterior (AP)
        x (axis 2): Patient Right -> Patient Left (RL)
    """

    data: np.ndarray
    spacing_mm: tuple[float, float, float]  # (dz, dy, dx) in mm
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame_of_reference_uid: str = "UNSPECIFIED"
    units: str = "Gy"
    name: str = "volume"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 3:
            raise ValueError(f"Volume3D '{self.name}' must be 3D, got shape {self.data.shape}")
        if not np.all(np.isfinite(self.data)):
            raise ValueError(f"Volume3D '{self.name}' contains NaN or Inf values")
        if len(self.spacing_mm) != 3 or any(s <= 0 for s in self.spacing_mm):
            raise ValueError(f"Invalid spacing_mm={self.spacing_mm}; all values must be > 0")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]), int(self.data.shape[2]))

    @property
    def voxel_volume_cc(self) -> float:
        """Single voxel volume in cubic centimeters (cm^3 = mL)."""
        dz, dy, dx = self.spacing_mm
        return float((dz * dy * dx) / 1000.0)

    def assert_compatible_grid(self, other: Volume3D, tol_mm: float = 1e-3) -> None:
        """Verify two volumes share identical dimensions and voxel spacing."""
        if self.shape != other.shape:
            raise ValueError(
                f"Grid shape mismatch between '{self.name}' {self.shape} "
                f"and '{other.name}' {other.shape}"
            )
        if any(abs(a - b) > tol_mm for a, b in zip(self.spacing_mm, other.spacing_mm)):
            raise ValueError(
                f"Grid spacing mismatch between '{self.name}' {self.spacing_mm} "
                f"and '{other.name}' {other.spacing_mm}"
            )


class CourseSpec(BaseModel):
    """Metadata and radiobiological delivery parameters for one radiotherapy course."""

    course_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    modality: Modality = Modality.PHOTON
    model: RadiobiologyModel = RadiobiologyModel.LQ
    n_fractions: int = Field(..., ge=1, le=200)
    prescription_gy: float = Field(..., gt=0.0, le=200.0)
    months_before_reirt: float = Field(
        default=0.0,
        ge=0.0,
        description="Elapsed months between completion of this course and start of re-irradiation.",
    )
    frame_of_reference_uid: str = Field(..., min_length=1)
    # Optional regime-specific parameters
    d_t_gy: float | None = Field(
        default=None,
        gt=0.0,
        description="LQL transition dose per fraction d_T (Gy); defaults to OAR-specific value.",
    )
    fraction_duration_h: float = Field(
        default=0.0,
        ge=0.0,
        description="Single-fraction or continuous delivery duration (hours) for Lea-Catcheside G.",
    )
    fractions_per_day: int = Field(default=1, ge=1, le=4)
    interfraction_interval_h: float = Field(default=24.0, gt=0.0)
    fixed_rbe: float = Field(default=1.1, ge=0.8, le=3.5)
    mean_let_d_kev_um: float = Field(
        default=2.5,
        ge=0.0,
        le=30.0,
        description="Fallback dose-averaged LET (keV/um) if no 3D LET_d volume is provided.",
    )
    source_provenance: str = Field(default="DICOM-RT")

    @field_validator("frame_of_reference_uid")
    @classmethod
    def _reject_blank_uid(cls, v: str) -> str:
        if not v.strip() or v.strip().upper() == "UNSPECIFIED":
            raise ValueError("CourseSpec requires an explicit, non-empty frame_of_reference_uid")
        return v.strip()


class OARConstraint(BaseModel):
    """Sourced clinical OAR radiobiology and cumulative EQD2 constraint specification."""

    tg263_name: str
    organ_class: str = Field(..., pattern="^(serial|parallel|mixed)$")
    alpha_beta_gy: float = Field(..., gt=0.0, le=25.0)
    alpha_beta_low_gy: float = Field(..., gt=0.0, le=25.0)
    alpha_beta_high_gy: float = Field(..., gt=0.0, le=25.0)
    repair_half_time_h: float = Field(default=1.5, gt=0.0, le=10.0)
    lql_d_t_gy: float = Field(
        default=6.0,
        gt=1.0,
        le=20.0,
        description="LQL linear-transition dose/fx d_T (Gy) per Astrahan (2008).",
    )
    primary_metric: str = Field(
        default="D0.03cc",
        description="Primary evaluation metric: 'D0.03cc' (near-max serial) or 'Dmean' (parallel).",
    )
    eqd2_limit_gy: float = Field(..., gt=0.0, description="Cumulative EQD2 tolerance limit (Gy).")
    eqd2_caution_gy: float = Field(..., gt=0.0, description="Cumulative EQD2 caution threshold.")
    max_recovery_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Conservative maximum fractional tissue recovery allowed after >= 24 months.",
    )
    min_recovery_months: float = Field(
        default=6.0,
        ge=0.0,
        description="Minimum interval (months) before any tissue recovery credit is permitted.",
    )
    citation: str
    doi: str


class DIRQualityMetrics(BaseModel):
    """AAPM TG-132 deformable image registration quality & diffeomorphism diagnostics."""

    jacobian_min: float
    jacobian_p01: float
    jacobian_median: float
    jacobian_p99: float
    jacobian_max: float
    folded_voxel_count: int
    folded_voxel_fraction: float
    ice_mean_mm: float = Field(..., description="Mean Inverse-Consistency Error (mm).")
    ice_p95_mm: float = Field(..., description="95th-percentile Inverse-Consistency Error (mm).")
    ice_max_mm: float = Field(..., description="Maximum Inverse-Consistency Error (mm).")
    disp_mean_mm: float
    disp_max_mm: float
    ensemble_tre_sigma_mm: float
    ensemble_size: int
    tg132_pass: bool
    notes: list[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    """Immutable SHA-256 provenance and invariant verification record."""

    engine_version: str
    timestamp_utc: str
    case_id: str
    dataset_source: str
    dataset_citation: str
    dataset_license: str
    input_checksums: dict[str, str]
    invariants_checked: list[str]
    max_inverse_residual_gy: float
    all_invariants_passed: bool
    abstention_triggered: bool = False
    abstention_reason: str | None = None


class OARBudgetSummary(BaseModel):
    """Structured per-OAR cumulative EQD2 and remaining physical dose budget summary."""

    tg263_name: str
    organ_class: str
    metric_name: str
    volume_cc: float
    alpha_beta_gy: float
    recovery_fraction_applied: float
    # Prior Course 1 EQD2 (at Course 2 anatomy, after DIR envelope)
    prior_eqd2_unrecovered_nominal_gy: float
    prior_eqd2_unrecovered_p95_gy: float
    prior_eqd2_effective_p05_gy: float
    prior_eqd2_effective_nominal_gy: float
    prior_eqd2_effective_p95_gy: float
    prior_eqd2_effective_worst_gy: float
    # Limits and headroom
    eqd2_limit_gy: float
    eqd2_caution_gy: float
    remaining_eqd2_nominal_gy: float
    remaining_eqd2_p95_gy: float
    # Inverted remaining physical dose budget for the planned Course 2 schedule
    remaining_physical_nominal_gy: float
    remaining_physical_p95_gy: float
    remaining_physical_worst_gy: float
    # Optional evaluation if a candidate Course 2 dose is supplied
    planned_course2_eqd2_gy: float | None = None
    cumulative_eqd2_nominal_gy: float | None = None
    cumulative_eqd2_p95_gy: float | None = None
    verdict: AuditVerdict
    citation: str
    doi: str


class ReIrradiationResult(BaseModel):
    """Complete serializable output of a multi-course re-irradiation budget computation."""

    case_id: str
    estro_eortc_classification: ReIrradiationType
    global_verdict: AuditVerdict
    prior_courses: list[CourseSpec]
    planned_course: CourseSpec
    dir_quality: DIRQualityMetrics
    per_course_dir_quality: list[DIRQualityMetrics] = Field(default_factory=list)
    oar_summaries: list[OARBudgetSummary]
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)


@dataclass
class PriorCourseInput:
    """Bundled prior radiotherapy course volume, DVF, and registration uncertainty for K-course solving."""

    course: CourseSpec
    dose: Volume3D
    nominal_dvf_mm: np.ndarray
    nominal_inv_dvf_mm: np.ndarray | None = None
    tre_sigma_mm: float = 2.5
    let_d: Volume3D | None = None
    recovery_overrides: dict[str, float] | None = None


class LKBModelSpec(BaseModel):
    """Sourced Lyman-Kutcher-Burman (LKB) / gEUD Normal Tissue Complication Probability parameters."""

    tg263_name: str
    endpoint: str
    td50_gy: float = Field(..., gt=0.0)
    m: float = Field(..., gt=0.0, le=2.0)
    n: float = Field(..., gt=0.0, le=2.0)
    citation: str
    doi: str


class OARNTCPSummary(BaseModel):
    """Cumulative gEUD and LKB NTCP probability with 90% joint uncertainty confidence interval."""

    tg263_name: str
    endpoint: str
    geud_p05_gy: float
    geud_nominal_gy: float
    geud_p95_gy: float
    ntcp_p05_pct: float
    ntcp_nominal_pct: float
    ntcp_p95_pct: float
    ntcp_worst_pct: float
    td50_gy: float
    m: float
    n: float
    citation: str
    doi: str


class SobolSensitivityResult(BaseModel):
    """First-order (S_i), total-order (S_Ti), and interaction (S_Ti - S_i) Sobol indices."""

    oar_name: str
    output_metric: str
    n_base_samples: int
    total_evaluations: int
    parameter_names: list[str]
    first_order_s1: dict[str, float]
    total_order_st: dict[str, float]
    interaction_st_minus_s1: dict[str, float]
    output_mean: float
    output_std: float
    output_p05: float
    output_p50: float
    output_p95: float
    dominant_parameter: str


class SourceSpanEvidence(BaseModel):
    """Exact verbatim source span proving where a legacy clinical fact came from."""

    field_name: str
    value_extracted: str
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    quote: str


class LegacyDVHConstraint(BaseModel):
    """One sparse OAR dosimetric constraint extracted from a legacy PDF/text report."""

    tg263_name: str
    metric_name: str = Field(..., description="'Dmax', 'D0.03cc', 'Dmean', or 'V<x>Gy'")
    value: float = Field(..., ge=0.0, description="Dose in Gy (or volume % for V_xGy).")
    units: str = Field(default="Gy")
    source_span: SourceSpanEvidence | None = None


class LegacyCourseRecord(BaseModel):
    """Structured legacy prior-course summary parsed when prior DICOM-RT files are missing."""

    record_id: str
    raw_source_label: str = "Outside-Hospital PDF Summary (No DICOM-RT)"
    modality: Modality = Modality.PHOTON
    model: RadiobiologyModel = RadiobiologyModel.LQ
    beam_energy_label: str = "6 MV IMRT/VMAT"
    n_fractions: int = Field(..., ge=1, le=200)
    prescription_gy: float = Field(..., gt=0.0, le=200.0)
    global_dmax_gy: float | None = Field(default=None, gt=0.0)
    months_before_reirt: float = Field(default=18.0, ge=0.0)
    oar_constraints: list[LegacyDVHConstraint] = Field(default_factory=list)
    evidence_spans: list[SourceSpanEvidence] = Field(default_factory=list)


class LegacyOARReconstructionSummary(BaseModel):
    """Per-OAR comparison of Legacy-PDF reconstructed envelope vs. optional hidden DICOM truth."""

    tg263_name: str
    metric_name: str
    reported_legacy_dose_gy: float | None = None
    reconstructed_nom_dose_gy: float
    reconstructed_upper_dose_gy: float
    legacy_remaining_physical_ceiling_gy: float
    # Populated when validated against hidden ground-truth DICOM/OpenKBP 3D dose
    true_dicom_dose_gy: float | None = None
    true_dicom_remaining_physical_ceiling_gy: float | None = None
    dicom_recovery_value_gy: float | None = Field(
        default=None,
        description="Additional verified physical headroom (Gy) unlocked if original DICOM is retrieved.",
    )
    envelope_safely_bounds_truth: bool | None = None


class LegacyReconstructionAudit(BaseModel):
    """Complete serializable audit output of Missing-DICOM 3D Prior-Dose Envelope Reconstruction."""

    case_id: str
    legacy_record: LegacyCourseRecord
    planned_course: CourseSpec
    penumbra_gradient_max_gy_mm: float
    pocs_iterations: int
    max_constraint_fit_error_gy: float
    max_inverse_residual_gy: float
    oar_summaries: list[LegacyOARReconstructionSummary]
    # Ground-truth validation metrics (when blinded truth volume is provided)
    truth_voxel_upper_safety_rate_pct: float | None = None
    truth_voxel_band_coverage_pct: float | None = None
    provenance: ProvenanceRecord
    warnings: list[str] = Field(default_factory=list)

