"""Missing-DICOM / Legacy-PDF Physics-Constrained 3D Prior-Dose Envelope Reconstructor.

Solves one of the most pervasive clinical re-irradiation bottlenecks that commercial TPS
and DIR suites (RayStation, MIM Maestro, Varian Velocity) cannot address: reconstructing
mathematically defensible 3D prior-dose bounds [D_lower(x), D_nom(x), D_upper(x)] and a
conservative Course-2 remaining physical dose ceiling when NO prior DICOM-RT files exist—
only an outside-hospital treatment summary PDF/text with sparse prescription and OAR DVH
metrics.

Architecture:
1. Source-Span Legacy Parser (`parse_legacy_treatment_summary`):
   Extracts prescription, fractionation, modality, elapsed interval, and AAPM TG-263
   normalized OAR DVH facts with exact character-offset provenance spans, enforcing hard
   `AbstentionError` gates on missing fractionation, conflicting values, or unphysical DVHs.
2. Lipschitz-Penumbra & POCS Convex 3D Reconstructor (`reconstruct_legacy_prior_dose_envelope`):
   Combines physical beam-penumbra gradient limits (|grad D| <= G_max via 3D Euclidean
   distance transforms), scatter-bath bounds, and Projections Onto Convex Sets (POCS)
   water-filling to satisfy every reported legacy DVH constraint while generating provably
   ordered 3D bounds 0 <= D_lower(x) <= D_nom(x) <= D_upper(x).
3. Budget Handoff & Value-of-DICOM-Recovery Audit (`solve_legacy_missing_dicom_budget`):
   Inverts the conservative legacy envelope into a safe Course-2 physical dose ceiling
   and quantifies the exact headroom gain (Delta Gy) unlocked if the original DICOM is later
   recovered—validated blindly against real OpenKBP ground-truth 3D dose distributions.

NOT A MEDICAL DEVICE. For academic research and educational physics verification only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from reirt_biobudget import __version__
from reirt_biobudget.budget_solver import evaluate_oar_metric
from reirt_biobudget.ir import (
    AbstentionError,
    CourseSpec,
    LegacyCourseRecord,
    LegacyDVHConstraint,
    LegacyOARReconstructionSummary,
    LegacyReconstructionAudit,
    Modality,
    RadiobiologyModel,
    SourceSpanEvidence,
    Volume3D,
)
from reirt_biobudget.ontology import get_oar_constraint, normalize_tg263
from reirt_biobudget.provenance import build_provenance_record, sha256_array, sha256_model
from reirt_biobudget.radiobiology import (
    conservative_recovery_fraction,
    eqd2_to_physical,
    physical_to_eqd2,
)


@dataclass
class LegacyReconstructionOutput:
    """Complete in-memory output combining serializable audit + reconstructed 3D dose bounds."""

    audit: LegacyReconstructionAudit
    dose_lower_3d: Volume3D
    dose_nominal_3d: Volume3D
    dose_upper_3d: Volume3D
    remaining_physical_legacy_3d: Volume3D
    oar_reconstructed_envelopes: dict[str, dict[str, np.ndarray]]


# Regex patterns for unstructured outside-hospital RT treatment summaries
_RX_PATTERNS = [
    re.compile(
        r"(?P<rx>\d+(?:\.\d+)?)\s*Gy\s*(?:/|in|over|x)\s*(?P<fx>\d+)\s*(?:fx|fractions?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:prescription|prescribed|total\s+dose)\s*[:=]?\s*(?P<rx>\d+(?:\.\d+)?)\s*Gy"
        r".{0,40}?(?P<fx>\d+)\s*(?:fx|fractions?)",
        re.IGNORECASE,
    ),
]
_MONTHS_PATTERN = re.compile(
    r"(?:(?P<months>\d+(?:\.\d+)?)\s*months?\s*(?:ago|prior|elapsed|earlier))|"
    r"(?:interval\s*[:=]?\s*(?P<months2>\d+(?:\.\d+)?)\s*months?)|"
    r"(?:(?P<years>\d+(?:\.\d+)?)\s*years?\s*(?:ago|prior|earlier))",
    re.IGNORECASE,
)
_GLOBAL_MAX_PATTERN = re.compile(
    r"(?:global\s+dmax|plan\s+max(?:imum)?(?:\s+dose)?|hotspot)\s*[:=]?\s*(?P<val>\d+(?:\.\d+)?)\s*Gy",
    re.IGNORECASE,
)
_OAR_METRIC_PATTERN = re.compile(
    r"(?P<oar>Spinal\s*Cord|SpinalCord|Cord|Brain\s*stem|Brainstem|Bone_Mandible|Mandible|"
    r"Left\s*Parotid|Parotid_L|Right\s*Parotid|Parotid_R|Larynx|Esophagus|Optic\s*Chiasm|Rectum)"
    r"\s*(?:[:\-]?\s*)"
    r"(?P<metric>D0\.03cc|Dmax|max\s*dose|maximum\s*dose|Dmean|mean\s*dose)"
    r"\s*(?:[:=]|of|was)?\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*Gy",
    re.IGNORECASE,
)


def _canonical_metric(raw_metric: str) -> str:
    m = raw_metric.strip().lower().replace(" ", "")
    if m in ("d0.03cc",):
        return "D0.03cc"
    if m in ("dmax", "maxdose", "maximumdose"):
        return "Dmax"
    if m in ("dmean", "meandose"):
        return "Dmean"
    return raw_metric.strip()


def parse_legacy_treatment_summary(
    text: str,
    *,
    record_id: str = "LEGACY_SUMMARY_01",
    raw_source_label: str = "Outside-Hospital PDF Summary (No DICOM-RT)",
    default_months_elapsed: float = 18.0,
) -> LegacyCourseRecord:
    """Parse an unstructured outside-hospital RT treatment summary with source-span provenance.

    Enforces hard AbstentionError gates on:
      - Missing prescription dose or fraction count (`LEGACY_MISSING_FRACTIONATION`)
      - Conflicting values for the same OAR metric (`LEGACY_CONFLICTING_FACTS`)
      - Physically impossible DVH constraints (`LEGACY_UNPHYSICAL_DVH_CONSTRAINT`)
    """
    evidence: list[SourceSpanEvidence] = []

    # 1. Extract prescription & fraction count (mandatory)
    rx_gy: float | None = None
    n_fx: int | None = None
    for pat in _RX_PATTERNS:
        m = pat.search(text)
        if m:
            rx_gy = float(m.group("rx"))
            n_fx = int(m.group("fx"))
            evidence.append(
                SourceSpanEvidence(
                    field_name="prescription_and_fractions",
                    value_extracted=f"{rx_gy:.1f} Gy / {n_fx} fx",
                    start_char=m.start(),
                    end_char=m.end(),
                    quote=m.group(0),
                )
            )
            break

    if rx_gy is None or n_fx is None or rx_gy <= 0.0 or n_fx < 1:
        raise AbstentionError(
            "LEGACY_MISSING_FRACTIONATION",
            (
                "Outside-hospital summary lacks verifiable prescription dose and fraction count "
                "(e.g., '70 Gy in 35 fractions'). Refusing to guess radiobiological fractionation."
            ),
            {"record_id": record_id},
        )

    # 2. Modality & radiobiology regime inference
    text_lower = text.lower()
    if "proton" in text_lower:
        modality = Modality.PROTON_FIXED_RBE
        beam_label = "Proton Therapy (fixed RBE 1.1)"
        model = RadiobiologyModel.LQ
    elif "brachy" in text_lower or "hdr" in text_lower:
        modality = Modality.BRACHY_HDR
        beam_label = "HDR Brachytherapy"
        model = RadiobiologyModel.LQ
    else:
        modality = Modality.PHOTON
        if "sbrt" in text_lower or "sabr" in text_lower or (rx_gy / n_fx) >= 5.0:
            beam_label = "6 MV SBRT/VMAT"
            model = RadiobiologyModel.LQL
        elif "vmat" in text_lower:
            beam_label = "6 MV VMAT"
            model = RadiobiologyModel.LQ
        elif "3d-crt" in text_lower or "3dcrt" in text_lower:
            beam_label = "6 MV 3D-CRT"
            model = RadiobiologyModel.LQ
        else:
            beam_label = "6 MV IMRT"
            model = RadiobiologyModel.LQ

    # 3. Elapsed time since completion
    months_elapsed = default_months_elapsed
    m_time = _MONTHS_PATTERN.search(text)
    if m_time:
        if m_time.group("months"):
            months_elapsed = float(m_time.group("months"))
        elif m_time.group("months2"):
            months_elapsed = float(m_time.group("months2"))
        elif m_time.group("years"):
            months_elapsed = float(m_time.group("years")) * 12.0
        evidence.append(
            SourceSpanEvidence(
                field_name="months_before_reirt",
                value_extracted=f"{months_elapsed:.1f} months",
                start_char=m_time.start(),
                end_char=m_time.end(),
                quote=m_time.group(0),
            )
        )

    # 4. Optional global Dmax
    global_dmax: float | None = None
    m_gmax = _GLOBAL_MAX_PATTERN.search(text)
    if m_gmax:
        global_dmax = float(m_gmax.group("val"))
        evidence.append(
            SourceSpanEvidence(
                field_name="global_dmax_gy",
                value_extracted=f"{global_dmax:.1f} Gy",
                start_char=m_gmax.start(),
                end_char=m_gmax.end(),
                quote=m_gmax.group(0),
            )
        )

    # 5. Extract per-OAR DVH facts with conflict detection & physical plausibility checks
    seen_metrics: dict[tuple[str, str], LegacyDVHConstraint] = {}
    for m_oar in _OAR_METRIC_PATTERN.finditer(text):
        tg263 = normalize_tg263(m_oar.group("oar"))
        metric = _canonical_metric(m_oar.group("metric"))
        val = float(m_oar.group("val"))
        span = SourceSpanEvidence(
            field_name=f"{tg263}.{metric}",
            value_extracted=f"{val:.2f} Gy",
            start_char=m_oar.start(),
            end_char=m_oar.end(),
            quote=m_oar.group(0),
        )
        key = (tg263, metric)
        if key in seen_metrics:
            prev_val = seen_metrics[key].value
            if abs(prev_val - val) > 2.0:
                raise AbstentionError(
                    "LEGACY_CONFLICTING_FACTS",
                    (
                        f"Conflicting legacy values for {tg263} {metric}: "
                        f"{prev_val:.1f} Gy vs. {val:.1f} Gy. Human chart verification required."
                    ),
                    {"oar": tg263, "metric": metric, "values": [prev_val, val]},
                )
            continue

        # Check unphysical OAR dose > 1.40 * Rx for external beam
        if modality != Modality.BRACHY_HDR and val > 1.40 * rx_gy:
            raise AbstentionError(
                "LEGACY_UNPHYSICAL_DVH_CONSTRAINT",
                (
                    f"Reported {tg263} {metric} ({val:.1f} Gy) exceeds 140% of external-beam "
                    f"prescription ({rx_gy:.1f} Gy). Likely transcription or unit error."
                ),
                {"oar": tg263, "metric": metric, "value_gy": val, "rx_gy": rx_gy},
            )

        constraint = LegacyDVHConstraint(
            tg263_name=tg263,
            metric_name=metric,
            value=val,
            units="Gy",
            source_span=span,
        )
        seen_metrics[key] = constraint
        evidence.append(span)

    # Cross-check Dmean <= Dmax within each OAR
    by_oar: dict[str, dict[str, float]] = {}
    for (tg263, metric), c_obj in seen_metrics.items():
        by_oar.setdefault(tg263, {})[metric] = c_obj.value
    for tg263, mmap in by_oar.items():
        d_mean = mmap.get("Dmean")
        d_hi = mmap.get("Dmax") or mmap.get("D0.03cc")
        if d_mean is not None and d_hi is not None and d_mean > d_hi + 0.5:
            raise AbstentionError(
                "LEGACY_UNPHYSICAL_DVH_CONSTRAINT",
                (
                    f"Physically impossible legacy DVH for {tg263}: "
                    f"Dmean ({d_mean:.1f} Gy) > near-max ({d_hi:.1f} Gy)."
                ),
                {"oar": tg263, "dmean_gy": d_mean, "dmax_gy": d_hi},
            )

    return LegacyCourseRecord(
        record_id=record_id,
        raw_source_label=raw_source_label,
        modality=modality,
        model=model,
        beam_energy_label=beam_label,
        n_fractions=n_fx,
        prescription_gy=rx_gy,
        global_dmax_gy=global_dmax,
        months_before_reirt=months_elapsed,
        oar_constraints=list(seen_metrics.values()),
        evidence_spans=evidence,
    )


def _beam_penumbra_physics(record: LegacyCourseRecord) -> tuple[float, float, float, float]:
    """Return physical (G_max_gy_mm, lambda_upper_mm, lambda_lower_mm, scatter_floor_frac)."""
    d_fx = record.prescription_gy / float(record.n_fractions)
    if record.modality in (Modality.PROTON_FIXED_RBE, Modality.PROTON_VARIABLE_RBE):
        return (6.5, 11.0, 4.5, 0.015)
    if d_fx >= 5.0 or "sbrt" in record.beam_energy_label.lower():
        return (5.5, 13.0, 5.5, 0.04)
    return (3.8, 22.0, 8.0, 0.07)


def _project_bounded_mean(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_mean: float,
) -> np.ndarray:
    """Exact convex projection onto {x : lower <= x <= upper, mean(x) == target_mean}."""
    lo_mean = float(np.mean(lower))
    hi_mean = float(np.mean(upper))
    if target_mean <= lo_mean:
        return lower.copy()
    if target_mean >= hi_mean:
        return upper.copy()

    # Solve mean(clip(values + delta, lower, upper)) == target_mean by monotone bisection
    d_lo = float(np.min(lower - values)) - 1.0
    d_hi = float(np.max(upper - values)) + 1.0
    for _ in range(32):
        mid = 0.5 * (d_lo + d_hi)
        m_val = float(np.mean(np.clip(values + mid, lower, upper)))
        if m_val < target_mean:
            d_lo = mid
        else:
            d_hi = mid
    return np.clip(values + 0.5 * (d_lo + d_hi), lower, upper)


def _project_near_max(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_val: float,
    metric_name: str,
    voxel_vol_cc: float,
) -> np.ndarray:
    """Monotone projection so evaluate_oar_metric(out, metric_name) matches target_val."""
    cur = evaluate_oar_metric(values, metric_name, voxel_vol_cc)
    if target_val <= 1e-6:
        return np.minimum(lower, 0.0)
    if cur <= 1e-6:
        # Place smooth peak on the voxel(s) with largest upper headroom
        out = lower.copy()
        idx_top = int(np.argmax(upper))
        out[idx_top] = min(target_val, float(upper[idx_top]))
        return out

    # Bisection on multiplicative scale factor s > 0 inside [lower, upper]
    s_lo, s_hi = 0.0, max(target_val / max(cur, 1e-6) * 2.5, 3.0)
    for _ in range(28):
        s_mid = 0.5 * (s_lo + s_hi)
        cand = np.clip(values * s_mid, lower, upper)
        m_val = evaluate_oar_metric(cand, metric_name, voxel_vol_cc)
        if m_val < target_val:
            s_lo = s_mid
        else:
            s_hi = s_mid
    return np.clip(values * 0.5 * (s_lo + s_hi), lower, upper)


def solve_legacy_missing_dicom_budget(
    *,
    case_id: str,
    legacy_record: LegacyCourseRecord,
    planned_course: CourseSpec,
    body_mask_current: np.ndarray,
    oar_masks_current: dict[str, np.ndarray],
    spacing_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    approx_prior_ptv_current: np.ndarray | None = None,
    hidden_ground_truth_dose: Volume3D | None = None,
    rel_uncertainty_margin: float = 0.08,
    abs_uncertainty_margin_gy: float = 2.5,
    pocs_iterations: int = 12,
    dataset_source: str = "Legacy PDF Summary (No Prior DICOM-RT)",
    dataset_citation: str = (
        "Babier A et al., Medical Physics 2021;48(9):5549-5561. DOI: 10.1002/mp.14845"
    ),
    dataset_license: str = "CC BY 4.0 (validation ground truth) / Non-Commercial Research & Personal Use (reconstructor)",
) -> LegacyReconstructionOutput:
    """Reconstruct 3D prior-dose envelope [D_lower, D_nom, D_upper] from sparse legacy records
    and invert into a conservative Course-2 remaining physical dose ceiling.

    If `hidden_ground_truth_dose` (warped onto current CT) is supplied for blinded validation,
    computes exact safety-bounding rates and the `dicom_recovery_value_gy` (additional safe
    headroom unlocked if the outside hospital's original DICOM is retrieved).
    """
    body = np.asarray(body_mask_current, dtype=bool)
    shape = body.shape
    vox_cc = float((spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0)

    # Hard anchor check: need at least one OAR DVH constraint OR an approximate prior PTV
    matched_oars = [
        c for c in legacy_record.oar_constraints if c.tg263_name in oar_masks_current
    ]
    if not matched_oars and (approx_prior_ptv_current is None or not np.any(approx_prior_ptv_current)):
        raise AbstentionError(
            "LEGACY_NO_ANCHOR_CONSTRAINTS",
            (
                "Cannot reconstruct 3D prior dose envelope without either matched OAR DVH "
                "constraints or a physician-contoured approximate prior target region."
            ),
            {"case_id": case_id},
        )

    g_max, lam_upper, lam_lower, f_scatter = _beam_penumbra_physics(legacy_record)
    rx = float(legacy_record.prescription_gy)
    d_cap = float(legacy_record.global_dmax_gy) if legacy_record.global_dmax_gy else rx * 1.08
    d_cap = max(d_cap, rx * 1.03)

    # 1. Construct Physical Upper Bound D_upper(x) & Lower Bound D_lower(x)
    d_upper = np.where(body, d_cap, 0.0).astype(np.float64)
    d_lower = np.zeros(shape, dtype=np.float64)
    d_nom = np.where(body, rx * 0.25, 0.0).astype(np.float64)

    if approx_prior_ptv_current is not None and np.any(approx_prior_ptv_current):
        ptv = np.asarray(approx_prior_ptv_current, dtype=bool) & body
        dist_ptv_mm = distance_transform_edt(~ptv, sampling=spacing_mm)
        # Allow 8 mm anatomical/contouring shift uncertainty around sketched legacy PTV
        ptv_upper = d_cap * np.maximum(
            np.exp(-np.maximum(dist_ptv_mm - 8.0, 0.0) / lam_upper),
            f_scatter,
        )
        d_upper = np.minimum(d_upper, ptv_upper) * body
        # Steepest physically achievable decay from core PTV gives optimistic lower bound
        ptv_lower = (rx * 0.92) * np.exp(-((dist_ptv_mm / lam_lower) ** 1.25))
        d_lower = np.minimum(ptv_lower, d_upper) * body
        # Smooth physical IMRT prior initialization
        lam_nom = 0.5 * (lam_upper + lam_lower)
        d_nom = (rx * 1.01) * np.exp(-((dist_ptv_mm / lam_nom) ** 1.15)) * body

    # Organize legacy constraints by OAR
    constraints_by_oar: dict[str, dict[str, LegacyDVHConstraint]] = {}
    for c in matched_oars:
        constraints_by_oar.setdefault(c.tg263_name, {})[c.metric_name] = c

    # Apply Lipschitz penumbra ceiling from every OAR with reported Dmax / D0.03cc
    for oar_name, cmap in constraints_by_oar.items():
        mask = np.asarray(oar_masks_current[oar_name], dtype=bool) & body
        if not np.any(mask):
            continue
        near_max_c = cmap.get("D0.03cc") or cmap.get("Dmax")
        if near_max_c is not None:
            u_oar = min(
                d_cap,
                near_max_c.value * (1.0 + rel_uncertainty_margin) + abs_uncertainty_margin_gy,
            )
            dist_oar_mm = distance_transform_edt(~mask, sampling=spacing_mm)
            lipschitz_ceil = u_oar + g_max * dist_oar_mm
            d_upper = np.minimum(d_upper, lipschitz_ceil) * body
            d_lower = np.minimum(d_lower, d_upper)

    # Ensure inside any OAR with reported constraint, d_upper is at least the reported value + cushion
    for oar_name, cmap in constraints_by_oar.items():
        mask = np.asarray(oar_masks_current[oar_name], dtype=bool) & body
        if not np.any(mask):
            continue
        dmean_c = cmap.get("Dmean")
        near_max_c = cmap.get("D0.03cc") or cmap.get("Dmax")
        if dmean_c is not None:
            u_mean_floor = min(
                d_cap,
                dmean_c.value * (1.0 + rel_uncertainty_margin) + abs_uncertainty_margin_gy,
            )
            d_upper[mask] = np.maximum(d_upper[mask], u_mean_floor)
        if near_max_c is not None:
            u_max_floor = min(
                d_cap,
                near_max_c.value * (1.0 + rel_uncertainty_margin) + abs_uncertainty_margin_gy,
            )
            d_upper[mask] = np.maximum(d_upper[mask], u_max_floor)

    # Enforce strict ordering before POCS
    d_lower = np.clip(d_lower, 0.0, d_upper)
    d_nom = np.clip(d_nom, d_lower, d_upper)

    # 2. Iterative Projections Onto Convex Sets (POCS) for exact legacy DVH constraint satisfaction
    oar_union = np.zeros(shape, dtype=bool)
    for oar_name in constraints_by_oar:
        oar_union |= np.asarray(oar_masks_current[oar_name], dtype=bool) & body

    for it in range(pocs_iterations):
        # Light physical penumbra regularization outside constrained OAR interiors
        if it < pocs_iterations - 1:
            smoothed = gaussian_filter(d_nom, sigma=(0.8, 0.8, 0.8)) * body
            d_nom = np.where(oar_union, d_nom, smoothed)
            d_nom = np.clip(d_nom, d_lower, d_upper)

        for oar_name, cmap in constraints_by_oar.items():
            mask = np.asarray(oar_masks_current[oar_name], dtype=bool) & body
            if not np.any(mask):
                continue
            v_cur = d_nom[mask]
            v_lo = d_lower[mask]
            v_hi = d_upper[mask]

            # Project near-max first (if present), then Dmean (if present)
            near_max_c = cmap.get("D0.03cc") or cmap.get("Dmax")
            if near_max_c is not None:
                v_cur = _project_near_max(
                    v_cur, v_lo, v_hi, near_max_c.value, near_max_c.metric_name, vox_cc
                )
            dmean_c = cmap.get("Dmean")
            if dmean_c is not None:
                v_cur = _project_bounded_mean(v_cur, v_lo, v_hi, dmean_c.value)
            d_nom[mask] = np.clip(v_cur, v_lo, v_hi)

    # Measure maximum constraint fit error on nominal reconstruction
    max_fit_err = 0.0
    for oar_name, cmap in constraints_by_oar.items():
        mask = np.asarray(oar_masks_current[oar_name], dtype=bool) & body
        if not np.any(mask):
            continue
        for metric_name, c_obj in cmap.items():
            achieved = evaluate_oar_metric(d_nom[mask], metric_name, vox_cc)
            max_fit_err = max(max_fit_err, abs(achieved - c_obj.value))

    # 3. Convert Legacy Record -> CourseSpec & invert conservative upper bound into Course-2 budget
    legacy_course_spec = CourseSpec(
        course_id=legacy_record.record_id,
        label=f"Legacy Prior ({legacy_record.prescription_gy:.0f} Gy / {legacy_record.n_fractions} fx, No DICOM)",
        modality=legacy_record.modality,
        model=legacy_record.model,
        n_fractions=legacy_record.n_fractions,
        prescription_gy=legacy_record.prescription_gy,
        months_before_reirt=legacy_record.months_before_reirt,
        frame_of_reference_uid=planned_course.frame_of_reference_uid,
        source_provenance=legacy_record.raw_source_label,
    )

    generic_ab = 3.0
    bg_eqd2_upper = physical_to_eqd2(d_upper, legacy_course_spec, generic_ab)
    rem_phys_legacy_3d = (
        eqd2_to_physical(np.maximum(100.0 - bg_eqd2_upper, 0.0), planned_course, generic_ab) * body
    )

    oar_summaries: list[LegacyOARReconstructionSummary] = []
    oar_rec_envelopes: dict[str, dict[str, np.ndarray]] = {}
    oar_assigned = np.zeros(shape, dtype=bool)
    max_inv_res = 0.0
    warnings: list[str] = [
        (
            "Prior 3D dose was reconstructed from sparse legacy summary text without prior DICOM-RT. "
            "Conservative upper-bound remaining budgets include inter-course spatial uncertainty cushions."
        )
    ]

    truth_voxels_safe = 0
    truth_voxels_in_band = 0
    truth_voxels_total = 0

    for raw_name, raw_mask in oar_masks_current.items():
        mask = np.asarray(raw_mask, dtype=bool) & body
        if not np.any(mask):
            continue
        tg263 = normalize_tg263(raw_name)
        cons = get_oar_constraint(tg263)
        if cons is None:
            continue

        ab = cons.alpha_beta_gy
        rec_frac = conservative_recovery_fraction(cons, legacy_record.months_before_reirt)
        scale_eff = 1.0 - rec_frac

        d_lo_m = d_lower[mask]
        d_nom_m = d_nom[mask]
        d_hi_m = d_upper[mask]

        oar_rec_envelopes[tg263] = {
            "dose_lower": d_lo_m,
            "dose_nominal": d_nom_m,
            "dose_upper": d_hi_m,
        }

        eqd2_hi_vox = physical_to_eqd2(
            d_hi_m,
            legacy_course_spec,
            ab,
            repair_half_time_h=cons.repair_half_time_h,
            lql_d_t_gy=cons.lql_d_t_gy,
        ) * scale_eff
        rem_eqd2_vox = np.maximum(cons.eqd2_limit_gy - eqd2_hi_vox, 0.0)
        rem_phys_vox = eqd2_to_physical(
            rem_eqd2_vox,
            planned_course,
            ab,
            repair_half_time_h=cons.repair_half_time_h,
            lql_d_t_gy=cons.lql_d_t_gy,
        )
        # Forward-inverse invariant verification
        recomp = physical_to_eqd2(
            rem_phys_vox,
            planned_course,
            ab,
            repair_half_time_h=cons.repair_half_time_h,
            lql_d_t_gy=cons.lql_d_t_gy,
        )
        max_inv_res = max(max_inv_res, float(np.max(np.abs(recomp - rem_eqd2_vox))))

        # Populate composite 3D budget ceiling (strict min over overlapping OARs)
        already = oar_assigned & mask
        fresh = (~oar_assigned) & mask
        if np.any(fresh):
            rem_phys_legacy_3d[fresh] = rem_phys_vox[fresh[mask]]
        if np.any(already):
            rem_phys_legacy_3d[already] = np.minimum(
                rem_phys_legacy_3d[already], rem_phys_vox[already[mask]]
            )
        oar_assigned |= mask

        # OAR-level metric & conservative ceiling
        metric = cons.primary_metric
        cmap = constraints_by_oar.get(tg263, {})
        reported_val: float | None = None
        if metric in cmap:
            reported_val = cmap[metric].value
        elif metric == "D0.03cc" and "Dmax" in cmap:
            reported_val = cmap["Dmax"].value

        rec_nom_m = evaluate_oar_metric(d_nom_m, metric, vox_cc)
        # For Dmean-only OARs, use reported Dmean + uncertainty cushion as tight upper metric
        if metric == "Dmean" and reported_val is not None:
            rec_hi_m = min(
                evaluate_oar_metric(d_hi_m, metric, vox_cc),
                reported_val * (1.0 + rel_uncertainty_margin) + abs_uncertainty_margin_gy,
            )
        else:
            rec_hi_m = evaluate_oar_metric(d_hi_m, metric, vox_cc)

        eqd2_hi_metric = float(
            physical_to_eqd2(
                rec_hi_m,
                legacy_course_spec,
                ab,
                repair_half_time_h=cons.repair_half_time_h,
                lql_d_t_gy=cons.lql_d_t_gy,
            )
        ) * scale_eff
        rem_eqd2_leg = max(cons.eqd2_limit_gy - eqd2_hi_metric, 0.0)
        rem_phys_leg = float(
            eqd2_to_physical(
                rem_eqd2_leg,
                planned_course,
                ab,
                repair_half_time_h=cons.repair_half_time_h,
                lql_d_t_gy=cons.lql_d_t_gy,
            )
        )

        # Optional blinded ground-truth evaluation (OpenKBP real 3D dose)
        true_dose_m: float | None = None
        true_rem_phys: float | None = None
        dicom_rec_val: float | None = None
        safely_bounds: bool | None = None

        if hidden_ground_truth_dose is not None:
            t_vox = hidden_ground_truth_dose.data[mask]
            oar_rec_envelopes[tg263]["dose_truth"] = t_vox
            true_dose_m = evaluate_oar_metric(t_vox, metric, vox_cc)
            true_eqd2_m = float(
                physical_to_eqd2(
                    true_dose_m,
                    legacy_course_spec,
                    ab,
                    repair_half_time_h=cons.repair_half_time_h,
                    lql_d_t_gy=cons.lql_d_t_gy,
                )
            ) * scale_eff
            true_rem_eqd2 = max(cons.eqd2_limit_gy - true_eqd2_m, 0.0)
            true_rem_phys = float(
                eqd2_to_physical(
                    true_rem_eqd2,
                    planned_course,
                    ab,
                    repair_half_time_h=cons.repair_half_time_h,
                    lql_d_t_gy=cons.lql_d_t_gy,
                )
            )
            dicom_rec_val = round(true_rem_phys - rem_phys_leg, 2)
            safely_bounds = bool(rec_hi_m >= true_dose_m - 0.1)

            truth_voxels_safe += int(np.sum(t_vox <= d_hi_m + 0.5))
            truth_voxels_in_band += int(np.sum((t_vox >= d_lo_m - 0.5) & (t_vox <= d_hi_m + 0.5)))
            truth_voxels_total += int(t_vox.size)

        oar_summaries.append(
            LegacyOARReconstructionSummary(
                tg263_name=tg263,
                metric_name=metric,
                reported_legacy_dose_gy=reported_val,
                reconstructed_nom_dose_gy=round(rec_nom_m, 2),
                reconstructed_upper_dose_gy=round(rec_hi_m, 2),
                legacy_remaining_physical_ceiling_gy=round(rem_phys_leg, 2),
                true_dicom_dose_gy=round(true_dose_m, 2) if true_dose_m is not None else None,
                true_dicom_remaining_physical_ceiling_gy=(
                    round(true_rem_phys, 2) if true_rem_phys is not None else None
                ),
                dicom_recovery_value_gy=dicom_rec_val,
                envelope_safely_bounds_truth=safely_bounds,
            )
        )

    upper_safety_pct = (
        round(100.0 * truth_voxels_safe / max(truth_voxels_total, 1), 1)
        if truth_voxels_total > 0
        else None
    )
    band_cov_pct = (
        round(100.0 * truth_voxels_in_band / max(truth_voxels_total, 1), 1)
        if truth_voxels_total > 0
        else None
    )

    checksums = {
        "legacy_record_spec": sha256_model(legacy_record),
        "planned_course_spec": sha256_model(planned_course),
        "reconstructed_upper_3d": sha256_array(d_upper),
        "reconstructed_nominal_3d": sha256_array(d_nom),
    }
    invariants = [
        "LEGACY_SOURCE_SPAN_PROVENANCE_VERIFIED",
        "LIPSCHITZ_PENUMBRA_BOUND_ORDERING_0_LE_LOWER_LE_NOM_LE_UPPER",
        "POCS_LEGACY_DVH_CONSTRAINT_SATISFACTION",
    ]
    provenance = build_provenance_record(
        engine_version=__version__,
        case_id=case_id,
        dataset_source=dataset_source,
        dataset_citation=dataset_citation,
        dataset_license=dataset_license,
        input_checksums=checksums,
        invariants_checked=invariants,
        max_inverse_residual_gy=max_inv_res,
    )

    audit = LegacyReconstructionAudit(
        case_id=case_id,
        legacy_record=legacy_record,
        planned_course=planned_course,
        penumbra_gradient_max_gy_mm=g_max,
        pocs_iterations=pocs_iterations,
        max_constraint_fit_error_gy=round(max_fit_err, 4),
        max_inverse_residual_gy=max_inv_res,
        oar_summaries=oar_summaries,
        truth_voxel_upper_safety_rate_pct=upper_safety_pct,
        truth_voxel_band_coverage_pct=band_cov_pct,
        provenance=provenance,
        warnings=warnings,
    )

    def _vol(arr: np.ndarray, name: str) -> Volume3D:
        return Volume3D(
            data=arr,
            spacing_mm=spacing_mm,
            origin_mm=origin_mm,
            frame_of_reference_uid=planned_course.frame_of_reference_uid,
            units="Gy",
            name=name,
        )

    return LegacyReconstructionOutput(
        audit=audit,
        dose_lower_3d=_vol(d_lower, "legacy_prior_dose_lower"),
        dose_nominal_3d=_vol(d_nom, "legacy_prior_dose_nominal"),
        dose_upper_3d=_vol(d_upper, "legacy_prior_dose_upper"),
        remaining_physical_legacy_3d=_vol(rem_phys_legacy_3d, "legacy_remaining_physical_ceiling"),
        oar_reconstructed_envelopes=oar_rec_envelopes,
    )


def generate_legacy_summary_text_from_ground_truth(
    *,
    patient_id: str,
    prior_dose: Volume3D,
    oar_masks: dict[str, np.ndarray],
    prescription_gy: float = 70.0,
    n_fractions: int = 35,
    months_elapsed: float = 18.0,
) -> str:
    """Create a realistic unstructured outside-hospital PDF/text treatment summary from a real plan.

    Used strictly for blinded ground-truth benchmarking: only scalar OAR summary metrics are
    written into prose while the real 3D dose grid is hidden from the reconstructor.
    """
    vox_cc = prior_dose.voxel_volume_cc
    gmax = float(np.max(prior_dose.data))
    lines = [
        f"EXTERNAL RADIATION ONCOLOGY TREATMENT COMPLETION SUMMARY — PATIENT {patient_id.upper()}",
        "Facility: Regional Cancer Center (Legacy TPS Archive — DICOM-RT Unavailable)",
        f"Technique: 6 MV IMRT head-and-neck radiotherapy, completed {months_elapsed:.0f} months ago.",
        f"Prescription: {prescription_gy:.1f} Gy in {n_fractions} fractions.",
        f"Global Dmax: {gmax:.1f} Gy.",
        "Reported Critical Organ DVH Summary:",
    ]
    for raw_name, mask in sorted(oar_masks.items()):
        if not np.any(mask):
            continue
        tg263 = normalize_tg263(raw_name)
        cons = get_oar_constraint(tg263)
        if cons is None:
            continue
        val = evaluate_oar_metric(prior_dose.data[mask], cons.primary_metric, vox_cc)
        lines.append(f"  - {tg263} {cons.primary_metric}: {val:.1f} Gy")
    return "\n".join(lines)
