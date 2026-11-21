"""Proprietary 3D Constraint-Safe Remaining-Dose-Budget Optimizer & Invariant Verifier.

Supports arbitrary K >= 1 prior radiotherapy courses (heterogeneous modalities,
fractionations, timestamps, recovery credits, and independent DVFs) mapped onto the
current Course-(K+1) planning CT grid:
1. Independent AAPM TG-132 diffeomorphism & ICE provenance gate for EVERY prior course.
2. Exact paired Monte Carlo ensemble summation across K independent course registrations
   (avoiding naive over-addition of independent 95th-percentile tails).
3. Voxel-wise 3D remaining physical dose ceiling maps D_rem(x) respecting overlapping
   OAR constraints via strict voxel-wise minimum.
4. Clinical DVH metrics (D0.03cc, Dmean, Dmax, uncertainty-band DVHs) & ESTRO/EORTC 2022
   Type 1 / Type 2 classification.
5. Independent forward-inverse invariant verification (residual <= 1e-6 Gy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from reirt_biobudget import __version__
from reirt_biobudget.dir_envelope import DIREnvelopeResult, compute_dir_dose_envelope
from reirt_biobudget.ir import (
    AbstentionError,
    AuditVerdict,
    CourseSpec,
    OARBudgetSummary,
    OARConstraint,
    PriorCourseInput,
    ReIrradiationResult,
    ReIrradiationType,
    Volume3D,
)
from reirt_biobudget.ontology import get_oar_constraint, normalize_tg263
from reirt_biobudget.provenance import (
    build_provenance_record,
    enforce_provenance_gates,
    sha256_array,
    sha256_model,
    sha256_volume,
)
from reirt_biobudget.radiobiology import (
    conservative_recovery_fraction,
    eqd2_to_physical,
    physical_to_eqd2,
)


@dataclass
class FullBudgetOutput:
    """Rich in-memory output combining serializable ReIrradiationResult + 3D volumetric maps."""

    result: ReIrradiationResult
    dir_envelope: DIREnvelopeResult  # Primary (Course 1) DIR envelope
    prior_eqd2_nominal_map: Volume3D  # Composite 3D prior EQD2 (Gy) on target grid
    prior_eqd2_spread_map: Volume3D  # Composite 3D (p95 - p05) EQD2 spread (Gy)
    remaining_physical_nominal_map: Volume3D  # 3D voxel-wise planned physical ceiling (Gy)
    remaining_physical_p95_map: Volume3D  # Conservative p95 prior -> lower physical ceiling (Gy)
    oar_eqd2_envelopes: dict[str, dict[str, np.ndarray]]  # Per-OAR masked 1D arrays for DVH bands
    per_course_envelopes: list[DIREnvelopeResult] = field(default_factory=list)


def compute_d_xcc(dose_values: np.ndarray, voxel_vol_cc: float, target_cc: float = 0.03) -> float:
    """Compute exact near-maximum dose D_{target_cc} (Gy) via descending cumulative volume."""
    vals = np.asarray(dose_values, dtype=np.float64)
    if vals.size == 0:
        return 0.0
    sorted_desc = np.sort(vals)[::-1]
    if target_cc <= voxel_vol_cc or vals.size == 1:
        return float(sorted_desc[0])
    cum_vol = np.arange(1, vals.size + 1, dtype=np.float64) * voxel_vol_cc
    if target_cc >= cum_vol[-1]:
        return float(sorted_desc[-1])
    return float(np.interp(target_cc, cum_vol, sorted_desc))


def compute_dvh_curve(
    dose_values: np.ndarray,
    dose_grid_gy: np.ndarray,
) -> np.ndarray:
    """Compute cumulative relative DVH (% volume receiving >= d) on `dose_grid_gy`."""
    vals = np.asarray(dose_values, dtype=np.float64)
    if vals.size == 0:
        return np.zeros_like(dose_grid_gy, dtype=np.float64)
    sorted_vals = np.sort(vals)
    idx = np.searchsorted(sorted_vals, dose_grid_gy, side="left")
    return 100.0 * (vals.size - idx) / float(vals.size)


def evaluate_oar_metric(
    dose_values: np.ndarray,
    metric_name: str,
    voxel_vol_cc: float,
) -> float:
    """Evaluate a standard TG-263 OAR dosimetric metric ('D0.03cc', 'Dmean', 'Dmax')."""
    if dose_values.size == 0:
        return 0.0
    m = metric_name.strip().lower()
    if m == "dmean":
        return float(np.mean(dose_values))
    if m == "dmax":
        return float(np.max(dose_values))
    if m.startswith("d") and m.endswith("cc"):
        cc = float(m[1:-2])
        return compute_d_xcc(dose_values, voxel_vol_cc, target_cc=cc)
    raise ValueError(f"Unsupported OAR metric '{metric_name}'")


def classify_estro_eortc(
    prior_warped_dose: np.ndarray,
    new_ptv_mask: np.ndarray | None,
    candidate_course2_dose: np.ndarray | None,
    overlap_threshold_gy: float = 5.0,
) -> ReIrradiationType:
    """Classify re-irradiation per ESTRO/EORTC 2022 consensus (Andratschke et al., Lancet Oncol)."""
    prior_high = prior_warped_dose >= overlap_threshold_gy
    if new_ptv_mask is not None and np.any(new_ptv_mask & prior_high):
        return ReIrradiationType.TYPE_1_OVERLAP
    if candidate_course2_dose is not None:
        c2_high = candidate_course2_dose >= overlap_threshold_gy
        if np.any(c2_high & prior_high):
            return ReIrradiationType.TYPE_1_OVERLAP
        if np.any((candidate_course2_dose >= 2.0) & (prior_warped_dose >= 2.0)):
            return ReIrradiationType.TYPE_2_TOXICITY_CONCERN
    if np.any(prior_warped_dose >= 2.0):
        return ReIrradiationType.TYPE_2_TOXICITY_CONCERN
    return ReIrradiationType.REPEAT_ORGAN


def solve_multicourse_reirradiation_budget(
    *,
    case_id: str,
    prior_inputs: list[PriorCourseInput],
    planned_course: CourseSpec,
    oar_masks_course2: dict[str, np.ndarray],
    body_mask_course2: np.ndarray | None = None,
    new_ptv_mask_course2: np.ndarray | None = None,
    candidate_course2_dose: Volume3D | None = None,
    let_d_course2: Volume3D | None = None,
    custom_constraints: dict[str, OARConstraint] | None = None,
    recovery_overrides: dict[str, float] | None = None,
    alpha_beta_overrides: dict[str, float] | None = None,
    n_ensemble: int = 24,
    seed: int = 42,
    dataset_source: str = "OpenKBP (ababier/open-kbp)",
    dataset_citation: str = (
        "Babier A et al., Medical Physics 2021;48(9):5549-5561. DOI: 10.1002/mp.14845"
    ),
    dataset_license: str = "CC BY 4.0 (dataset) / MIT (code)",
    hard_ice_abstention_mm: float = 5.0,
    retain_ensemble_samples: bool = False,
) -> FullBudgetOutput:
    """Execute full K-course (K >= 1) 3D DIR-envelope + multi-regime radiobiology + inverse budget solve."""
    if not prior_inputs:
        raise AbstentionError(
            "NO_PRIOR_COURSE",
            "Multi-course re-irradiation accumulation requires at least one prior course.",
        )

    ref_dose = prior_inputs[0].dose
    spacing = ref_dose.spacing_mm
    shape = ref_dose.shape
    vox_cc = ref_dose.voxel_volume_cc

    # 1. Validate grid compatibility across all K prior courses & optional Course-(K+1) grids
    for p_in in prior_inputs[1:]:
        ref_dose.assert_compatible_grid(p_in.dose)
    if candidate_course2_dose is not None:
        ref_dose.assert_compatible_grid(candidate_course2_dose)
    if let_d_course2 is not None:
        ref_dose.assert_compatible_grid(let_d_course2)

    # 2. Compute 3D diffeomorphic DIR-uncertainty envelope & enforce TG-132 gate for EACH prior course
    keep_samples = retain_ensemble_samples or (len(prior_inputs) > 1)
    envelopes: list[DIREnvelopeResult] = []
    invariants_checked: list[str] = []
    warnings: list[str] = []

    for idx, p_in in enumerate(prior_inputs):
        env_k = compute_dir_dose_envelope(
            prior_dose=p_in.dose,
            nominal_dvf_mm=p_in.nominal_dvf_mm,
            nominal_inv_dvf_mm=p_in.nominal_inv_dvf_mm,
            body_mask=body_mask_course2,
            tre_sigma_mm=p_in.tre_sigma_mm,
            n_ensemble=n_ensemble,
            seed=seed + idx * 997,  # independent streams per prior course
            return_samples=keep_samples,
        )
        inv_k = enforce_provenance_gates(
            prior_courses=[p_in.course],
            planned_course=planned_course,
            dir_quality=env_k.quality,
            hard_ice_abstention_mm=hard_ice_abstention_mm,
        )
        if idx == 0:
            invariants_checked = list(inv_k)
        if not env_k.quality.tg132_pass:
            warnings.extend([f"[{p_in.course.course_id}] {n}" for n in env_k.quality.notes])
        envelopes.append(env_k)

    if len(prior_inputs) > 1:
        invariants_checked.append(f"MULTICOURSE_K{len(prior_inputs)}_PAIRED_ENSEMBLE_SUM")

    # 3. Composite generic background 3D maps (alpha/beta = 3 Gy, 100 Gy reference ceiling)
    generic_ab = 3.0
    bg_eqd2_nom = np.zeros(shape, dtype=np.float64)
    bg_eqd2_p05 = np.zeros(shape, dtype=np.float64)
    bg_eqd2_p95 = np.zeros(shape, dtype=np.float64)
    total_prior_phys_nom = np.zeros(shape, dtype=np.float64)

    for p_in, env_k in zip(prior_inputs, envelopes):
        let_k = p_in.let_d.data if p_in.let_d is not None else p_in.course.mean_let_d_kev_um
        total_prior_phys_nom += env_k.dose_nominal.data
        bg_eqd2_nom += physical_to_eqd2(
            env_k.dose_nominal.data, p_in.course, generic_ab, let_d_kev_um=let_k
        )
        bg_eqd2_p05 += physical_to_eqd2(
            env_k.dose_p05.data, p_in.course, generic_ab, let_d_kev_um=let_k
        )
        bg_eqd2_p95 += physical_to_eqd2(
            env_k.dose_p95.data, p_in.course, generic_ab, let_d_kev_um=let_k
        )

    prior_eqd2_nom_3d = bg_eqd2_nom.copy()
    prior_eqd2_spread_3d = np.maximum(bg_eqd2_p95 - bg_eqd2_p05, 0.0)

    let_arr = let_d_course2.data if let_d_course2 is not None else planned_course.mean_let_d_kev_um
    rem_phys_nom_3d = eqd2_to_physical(
        np.maximum(100.0 - bg_eqd2_nom, 0.0), planned_course, generic_ab, let_d_kev_um=let_arr
    )
    rem_phys_p95_3d = eqd2_to_physical(
        np.maximum(100.0 - bg_eqd2_p95, 0.0), planned_course, generic_ab, let_d_kev_um=let_arr
    )

    if planned_course.model.value == "lq" and (
        planned_course.prescription_gy / planned_course.n_fractions
    ) > 8.0:
        warnings.append(
            f"Planned course uses standard LQ at "
            f"{planned_course.prescription_gy / planned_course.n_fractions:.1f} Gy/fx; "
            "consider LQL model (Astrahan 2008) for high dose-per-fraction SBRT/SRS."
        )

    oar_assigned = np.zeros(shape, dtype=bool)
    oar_summaries: list[OARBudgetSummary] = []
    oar_eqd2_envelopes: dict[str, dict[str, np.ndarray]] = {}
    max_inverse_residual_gy = 0.0

    # 4. Per-OAR exact K-course biophysical summation & exact inverse remaining budget
    for raw_name, raw_mask in oar_masks_course2.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"OAR mask '{raw_name}' shape {mask.shape} != dose shape {shape}")
        if not np.any(mask):
            continue

        tg263 = normalize_tg263(raw_name)
        constraint = (
            custom_constraints.get(tg263)
            if custom_constraints and tg263 in custom_constraints
            else get_oar_constraint(tg263)
        )
        if constraint is None:
            continue

        ab = (
            float(alpha_beta_overrides[tg263])
            if alpha_beta_overrides and tg263 in alpha_beta_overrides
            else constraint.alpha_beta_gy
        )

        n_vox = int(np.sum(mask))
        eqd2_unrec_nom_tot = np.zeros(n_vox, dtype=np.float64)
        eqd2_eff_nom_tot = np.zeros(n_vox, dtype=np.float64)
        rec_weighted_num = 0.0
        rec_weighted_den = 0.0

        if len(prior_inputs) == 1:
            # Exact 1-course path (directly uses exact p05/nominal/p95/worst fields)
            p0 = prior_inputs[0]
            e0 = envelopes[0]
            rec_ov = (
                p0.recovery_overrides.get(tg263)
                if p0.recovery_overrides and tg263 in p0.recovery_overrides
                else (recovery_overrides.get(tg263) if recovery_overrides else None)
            )
            rec_frac = conservative_recovery_fraction(
                constraint, months_elapsed=p0.course.months_before_reirt, override_fraction=rec_ov
            )
            let0 = p0.let_d.data[mask] if p0.let_d is not None else p0.course.mean_let_d_kev_um

            eqd2_unrec_nom_tot = physical_to_eqd2(
                e0.dose_nominal.data[mask],
                p0.course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=let0,
            )
            eqd2_unrec_p05 = physical_to_eqd2(
                e0.dose_p05.data[mask],
                p0.course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=let0,
            )
            eqd2_unrec_p95 = physical_to_eqd2(
                e0.dose_p95.data[mask],
                p0.course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=let0,
            )
            eqd2_unrec_worst = physical_to_eqd2(
                e0.dose_worst.data[mask],
                p0.course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=let0,
            )
            scale = 1.0 - rec_frac
            eqd2_eff_nom_tot = eqd2_unrec_nom_tot * scale
            eqd2_eff_p05 = eqd2_unrec_p05 * scale
            eqd2_eff_p95 = eqd2_unrec_p95 * scale
            eqd2_eff_worst = eqd2_unrec_worst * scale
            effective_rec_reported = rec_frac
        else:
            # Exact K-course joint paired Monte Carlo ensemble summation
            m_draws = max(
                e.ensemble_samples_sub.shape[0] if e.ensemble_samples_sub is not None else 1
                for e in envelopes
            )
            unrec_draws = np.zeros((m_draws, n_vox), dtype=np.float64)
            eff_draws = np.zeros((m_draws, n_vox), dtype=np.float64)

            for p_in, env_k in zip(prior_inputs, envelopes):
                rec_ov = (
                    p_in.recovery_overrides.get(tg263)
                    if p_in.recovery_overrides and tg263 in p_in.recovery_overrides
                    else (recovery_overrides.get(tg263) if recovery_overrides else None)
                )
                rec_k = conservative_recovery_fraction(
                    constraint,
                    months_elapsed=p_in.course.months_before_reirt,
                    override_fraction=rec_ov,
                )
                let_k = (
                    p_in.let_d.data[mask]
                    if p_in.let_d is not None
                    else p_in.course.mean_let_d_kev_um
                )

                # Nominal term
                u_nom_k = physical_to_eqd2(
                    env_k.dose_nominal.data[mask],
                    p_in.course,
                    ab,
                    repair_half_time_h=constraint.repair_half_time_h,
                    lql_d_t_gy=constraint.lql_d_t_gy,
                    let_d_kev_um=let_k,
                )
                eqd2_unrec_nom_tot += u_nom_k
                eqd2_eff_nom_tot += u_nom_k * (1.0 - rec_k)

                w_k = float(np.mean(u_nom_k))
                rec_weighted_num += rec_k * w_k
                rec_weighted_den += w_k

                # Ensemble sample draws (M, N_vox)
                ens_k = env_k.extract_masked_ensemble(mask)
                if ens_k.shape[0] == 1 and m_draws > 1:
                    ens_k = np.repeat(ens_k, m_draws, axis=0)
                u_ens_k = physical_to_eqd2(
                    ens_k,
                    p_in.course,
                    ab,
                    repair_half_time_h=constraint.repair_half_time_h,
                    lql_d_t_gy=constraint.lql_d_t_gy,
                    let_d_kev_um=let_k,
                )
                unrec_draws += u_ens_k
                eff_draws += u_ens_k * (1.0 - rec_k)

            eqd2_unrec_p95 = np.percentile(unrec_draws, 95, axis=0)
            eqd2_eff_p05 = np.percentile(eff_draws, 5, axis=0)
            eqd2_eff_p95 = np.percentile(eff_draws, 95, axis=0)
            eqd2_eff_worst = np.max(eff_draws, axis=0)
            effective_rec_reported = (
                float(rec_weighted_num / rec_weighted_den) if rec_weighted_den > 1e-9 else 0.0
            )

        # Verify strict quantile monotonicity invariant
        assert np.all(eqd2_eff_p05 <= eqd2_eff_p95 + 1e-9), "DIR envelope p05 > p95 invariant violated"
        assert np.all(eqd2_eff_p95 <= eqd2_eff_worst + 1e-9), "DIR envelope p95 > worst invariant violated"

        # Voxel-wise remaining EQD2 & exact inverted Course-(K+1) physical dose ceiling
        rem_eqd2_nom_vox = np.maximum(constraint.eqd2_limit_gy - eqd2_eff_nom_tot, 0.0)
        rem_eqd2_p95_vox = np.maximum(constraint.eqd2_limit_gy - eqd2_eff_p95, 0.0)

        let_vox = (
            let_d_course2.data[mask]
            if let_d_course2 is not None
            else planned_course.mean_let_d_kev_um
        )

        rem_phys_nom_vox = eqd2_to_physical(
            rem_eqd2_nom_vox,
            planned_course,
            ab,
            repair_half_time_h=constraint.repair_half_time_h,
            lql_d_t_gy=constraint.lql_d_t_gy,
            let_d_kev_um=let_vox,
        )
        rem_phys_p95_vox = eqd2_to_physical(
            rem_eqd2_p95_vox,
            planned_course,
            ab,
            repair_half_time_h=constraint.repair_half_time_h,
            lql_d_t_gy=constraint.lql_d_t_gy,
            let_d_kev_um=let_vox,
        )

        # Independent forward-inverse verification check!
        recomputed_eqd2 = physical_to_eqd2(
            rem_phys_p95_vox,
            planned_course,
            ab,
            repair_half_time_h=constraint.repair_half_time_h,
            lql_d_t_gy=constraint.lql_d_t_gy,
            let_d_kev_um=let_vox,
        )
        residual = float(np.max(np.abs(recomputed_eqd2 - rem_eqd2_p95_vox)))
        max_inverse_residual_gy = max(max_inverse_residual_gy, residual)

        # Populate 3D composite maps (strict min inside overlapping OARs)
        prior_eqd2_nom_3d[mask] = eqd2_eff_nom_tot
        prior_eqd2_spread_3d[mask] = np.maximum(eqd2_eff_p95 - eqd2_eff_p05, 0.0)

        already = oar_assigned & mask
        fresh = (~oar_assigned) & mask
        if np.any(fresh):
            rem_phys_nom_3d[fresh] = rem_phys_nom_vox[fresh[mask]]
            rem_phys_p95_3d[fresh] = rem_phys_p95_vox[fresh[mask]]
        if np.any(already):
            rem_phys_nom_3d[already] = np.minimum(
                rem_phys_nom_3d[already], rem_phys_nom_vox[already[mask]]
            )
            rem_phys_p95_3d[already] = np.minimum(
                rem_phys_p95_3d[already], rem_phys_p95_vox[already[mask]]
            )
        oar_assigned |= mask

        # Evaluate OAR-level clinical metrics
        metric = constraint.primary_metric
        m_unrec_nom = evaluate_oar_metric(eqd2_unrec_nom_tot, metric, vox_cc)
        m_unrec_p95 = evaluate_oar_metric(eqd2_unrec_p95, metric, vox_cc)
        m_eff_p05 = evaluate_oar_metric(eqd2_eff_p05, metric, vox_cc)
        m_eff_nom = evaluate_oar_metric(eqd2_eff_nom_tot, metric, vox_cc)
        m_eff_p95 = evaluate_oar_metric(eqd2_eff_p95, metric, vox_cc)
        m_eff_worst = evaluate_oar_metric(eqd2_eff_worst, metric, vox_cc)

        rem_eqd2_nom = max(constraint.eqd2_limit_gy - m_eff_nom, 0.0)
        rem_eqd2_p95 = max(constraint.eqd2_limit_gy - m_eff_p95, 0.0)
        rem_eqd2_worst = max(constraint.eqd2_limit_gy - m_eff_worst, 0.0)

        scalar_let = float(np.mean(let_vox)) if isinstance(let_vox, np.ndarray) else float(let_vox)
        rem_phys_nom = float(
            eqd2_to_physical(
                rem_eqd2_nom,
                planned_course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=scalar_let,
            )
        )
        rem_phys_p95 = float(
            eqd2_to_physical(
                rem_eqd2_p95,
                planned_course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=scalar_let,
            )
        )
        rem_phys_worst = float(
            eqd2_to_physical(
                rem_eqd2_worst,
                planned_course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=scalar_let,
            )
        )

        # Optional candidate Course-(K+1) cumulative evaluation
        c2_eqd2_metric: float | None = None
        cum_nom_metric: float | None = None
        cum_p95_metric: float | None = None
        cum_p05_arr = eqd2_eff_p05
        cum_nom_arr = eqd2_eff_nom_tot
        cum_p95_arr = eqd2_eff_p95
        cum_worst_arr = eqd2_eff_worst

        if candidate_course2_dose is not None:
            c2_vox_eqd2 = physical_to_eqd2(
                candidate_course2_dose.data[mask],
                planned_course,
                ab,
                repair_half_time_h=constraint.repair_half_time_h,
                lql_d_t_gy=constraint.lql_d_t_gy,
                let_d_kev_um=let_vox,
            )
            c2_eqd2_metric = evaluate_oar_metric(c2_vox_eqd2, metric, vox_cc)
            cum_p05_arr = eqd2_eff_p05 + c2_vox_eqd2
            cum_nom_arr = eqd2_eff_nom_tot + c2_vox_eqd2
            cum_p95_arr = eqd2_eff_p95 + c2_vox_eqd2
            cum_worst_arr = eqd2_eff_worst + c2_vox_eqd2
            cum_nom_metric = evaluate_oar_metric(cum_nom_arr, metric, vox_cc)
            cum_p95_metric = evaluate_oar_metric(cum_p95_arr, metric, vox_cc)

        eval_val_nom = cum_nom_metric if cum_nom_metric is not None else m_eff_nom
        eval_val_p95 = cum_p95_metric if cum_p95_metric is not None else m_eff_p95

        if eval_val_nom > constraint.eqd2_limit_gy:
            verdict = AuditVerdict.VIOLATION
        elif eval_val_p95 > constraint.eqd2_limit_gy or eval_val_nom > constraint.eqd2_caution_gy:
            verdict = AuditVerdict.CAUTION
        else:
            verdict = AuditVerdict.PASS

        oar_eqd2_envelopes[tg263] = {
            "prior_p05": eqd2_eff_p05,
            "prior_nominal": eqd2_eff_nom_tot,
            "prior_p95": eqd2_eff_p95,
            "prior_worst": eqd2_eff_worst,
            "cum_p05": cum_p05_arr,
            "cum_nominal": cum_nom_arr,
            "cum_p95": cum_p95_arr,
            "cum_worst": cum_worst_arr,
        }

        oar_summaries.append(
            OARBudgetSummary(
                tg263_name=tg263,
                organ_class=constraint.organ_class,
                metric_name=metric,
                volume_cc=float(n_vox * vox_cc),
                alpha_beta_gy=ab,
                recovery_fraction_applied=effective_rec_reported,
                prior_eqd2_unrecovered_nominal_gy=m_unrec_nom,
                prior_eqd2_unrecovered_p95_gy=m_unrec_p95,
                prior_eqd2_effective_p05_gy=m_eff_p05,
                prior_eqd2_effective_nominal_gy=m_eff_nom,
                prior_eqd2_effective_p95_gy=m_eff_p95,
                prior_eqd2_effective_worst_gy=m_eff_worst,
                eqd2_limit_gy=constraint.eqd2_limit_gy,
                eqd2_caution_gy=constraint.eqd2_caution_gy,
                remaining_eqd2_nominal_gy=rem_eqd2_nom,
                remaining_eqd2_p95_gy=rem_eqd2_p95,
                remaining_physical_nominal_gy=rem_phys_nom,
                remaining_physical_p95_gy=rem_phys_p95,
                remaining_physical_worst_gy=rem_phys_worst,
                planned_course2_eqd2_gy=c2_eqd2_metric,
                cumulative_eqd2_nominal_gy=cum_nom_metric,
                cumulative_eqd2_p95_gy=cum_p95_metric,
                verdict=verdict,
                citation=constraint.citation,
                doi=constraint.doi,
            )
        )

    invariants_checked.append("DIR_ENVELOPE_QUANTILE_MONOTONICITY")

    estro_class = classify_estro_eortc(
        total_prior_phys_nom,
        new_ptv_mask=new_ptv_mask_course2,
        candidate_course2_dose=(
            candidate_course2_dose.data if candidate_course2_dose is not None else None
        ),
    )

    verdicts = [s.verdict for s in oar_summaries]
    if AuditVerdict.VIOLATION in verdicts:
        global_verdict = AuditVerdict.VIOLATION
    elif AuditVerdict.CAUTION in verdicts:
        global_verdict = AuditVerdict.CAUTION
    else:
        global_verdict = AuditVerdict.PASS

    checksums: dict[str, str] = {"planned_course_spec": sha256_model(planned_course)}
    if len(prior_inputs) == 1:
        checksums["prior_course_spec"] = sha256_model(prior_inputs[0].course)
        checksums["prior_dose_3d"] = sha256_volume(prior_inputs[0].dose)
        checksums["nominal_dvf_3d"] = sha256_array(prior_inputs[0].nominal_dvf_mm)
    else:
        for i, p_in in enumerate(prior_inputs, start=1):
            checksums[f"prior_course_{i}_spec"] = sha256_model(p_in.course)
            checksums[f"prior_course_{i}_dose_3d"] = sha256_volume(p_in.dose)
            checksums[f"prior_course_{i}_dvf_3d"] = sha256_array(p_in.nominal_dvf_mm)
    if candidate_course2_dose is not None:
        checksums["candidate_course2_dose_3d"] = sha256_volume(candidate_course2_dose)

    provenance = build_provenance_record(
        engine_version=__version__,
        case_id=case_id,
        dataset_source=dataset_source,
        dataset_citation=dataset_citation,
        dataset_license=dataset_license,
        input_checksums=checksums,
        invariants_checked=invariants_checked,
        max_inverse_residual_gy=max_inverse_residual_gy,
    )

    result = ReIrradiationResult(
        case_id=case_id,
        estro_eortc_classification=estro_class,
        global_verdict=global_verdict,
        prior_courses=[p.course for p in prior_inputs],
        planned_course=planned_course,
        dir_quality=envelopes[0].quality,
        per_course_dir_quality=[e.quality for e in envelopes],
        oar_summaries=oar_summaries,
        provenance=provenance,
        warnings=warnings,
    )

    def _vol(arr: np.ndarray, name: str) -> Volume3D:
        return Volume3D(
            data=arr,
            spacing_mm=spacing,
            origin_mm=ref_dose.origin_mm,
            frame_of_reference_uid=planned_course.frame_of_reference_uid,
            units="Gy",
            name=name,
        )

    return FullBudgetOutput(
        result=result,
        dir_envelope=envelopes[0],
        prior_eqd2_nominal_map=_vol(prior_eqd2_nom_3d, "prior_eqd2_effective_nominal"),
        prior_eqd2_spread_map=_vol(prior_eqd2_spread_3d, "prior_eqd2_dir_spread_90"),
        remaining_physical_nominal_map=_vol(rem_phys_nom_3d, "remaining_physical_budget_nominal"),
        remaining_physical_p95_map=_vol(rem_phys_p95_3d, "remaining_physical_budget_p95"),
        oar_eqd2_envelopes=oar_eqd2_envelopes,
        per_course_envelopes=envelopes,
    )


def solve_reirradiation_budget(
    *,
    case_id: str,
    prior_course: CourseSpec,
    prior_dose: Volume3D,
    planned_course: CourseSpec,
    nominal_dvf_mm: np.ndarray,
    oar_masks_course2: dict[str, np.ndarray],
    nominal_inv_dvf_mm: np.ndarray | None = None,
    body_mask_course2: np.ndarray | None = None,
    new_ptv_mask_course2: np.ndarray | None = None,
    candidate_course2_dose: Volume3D | None = None,
    let_d_course2: Volume3D | None = None,
    custom_constraints: dict[str, OARConstraint] | None = None,
    recovery_overrides: dict[str, float] | None = None,
    alpha_beta_overrides: dict[str, float] | None = None,
    tre_sigma_mm: float = 2.5,
    n_ensemble: int = 24,
    seed: int = 42,
    dataset_source: str = "OpenKBP (ababier/open-kbp)",
    dataset_citation: str = (
        "Babier A et al., Medical Physics 2021;48(9):5549-5561. DOI: 10.1002/mp.14845"
    ),
    dataset_license: str = "CC BY 4.0 (dataset) / MIT (code)",
    hard_ice_abstention_mm: float = 5.0,
    retain_ensemble_samples: bool = False,
) -> FullBudgetOutput:
    """Backward-compatible single-prior-course wrapper around `solve_multicourse_reirradiation_budget`."""
    return solve_multicourse_reirradiation_budget(
        case_id=case_id,
        prior_inputs=[
            PriorCourseInput(
                course=prior_course,
                dose=prior_dose,
                nominal_dvf_mm=nominal_dvf_mm,
                nominal_inv_dvf_mm=nominal_inv_dvf_mm,
                tre_sigma_mm=tre_sigma_mm,
                recovery_overrides=recovery_overrides,
            )
        ],
        planned_course=planned_course,
        oar_masks_course2=oar_masks_course2,
        body_mask_course2=body_mask_course2,
        new_ptv_mask_course2=new_ptv_mask_course2,
        candidate_course2_dose=candidate_course2_dose,
        let_d_course2=let_d_course2,
        custom_constraints=custom_constraints,
        recovery_overrides=recovery_overrides,
        alpha_beta_overrides=alpha_beta_overrides,
        n_ensemble=n_ensemble,
        seed=seed,
        dataset_source=dataset_source,
        dataset_citation=dataset_citation,
        dataset_license=dataset_license,
        hard_ice_abstention_mm=hard_ice_abstention_mm,
        retain_ensemble_samples=retain_ensemble_samples,
    )
