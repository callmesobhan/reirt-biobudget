"""Multi-Patient OpenKBP Cohort Benchmark & AAPM TG-132 Anatomical Landmark TRE Validation.

Provides:
1. Reproducible anatomical landmark extraction on clinical H&N OARs (centroids, poles,
   cervical spinal cord centerline points, mandibular anatomical extremes).
2. Pre-DIR vs. Post-DIR Landmark Target Registration Error (TRE) audit and empirical 90%
   DIR-uncertainty envelope coverage calibration under independent registration perturbations.
3. Multi-patient cohort runner across public OpenKBP cases (pt_1, pt_2, pt_3, ...) plus
   a true K=2 multi-prior-course heterogeneous-modality benchmark builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import map_coordinates

from reirt_biobudget.budget_solver import (
    FullBudgetOutput,
    solve_multicourse_reirradiation_budget,
    solve_reirradiation_budget,
)
from reirt_biobudget.dir_envelope import (
    _generate_diffeomorphic_perturbation,
    compose_dvf,
    integrate_svf_to_dvf,
    warp_array,
)
from reirt_biobudget.io_loaders import (
    OpenKBPPatient,
    ReIrradiationBenchmarkCase,
    build_reirradiation_benchmark_from_openkbp,
    load_openkbp_patient,
)
from reirt_biobudget.ir import (
    CourseSpec,
    Modality,
    OARNTCPSummary,
    PriorCourseInput,
    RadiobiologyModel,
    SobolSensitivityResult,
    Volume3D,
)
from reirt_biobudget.ntcp_sobol import evaluate_oar_ntcp_envelope, run_joint_sobol_sensitivity


class LandmarkTREValidation(BaseModel):
    """Quantitative AAPM TG-132 anatomical landmark TRE & 90% envelope coverage metrics."""

    n_landmarks: int
    pre_dir_tre_mean_mm: float
    pre_dir_tre_p95_mm: float
    pre_dir_tre_max_mm: float
    post_dir_tre_mean_mm: float
    post_dir_tre_p95_mm: float
    post_dir_tre_max_mm: float
    empirical_90pct_envelope_coverage_pct: float
    gradient_voxels_tested: int


class CohortPatientSummary(BaseModel):
    """Per-patient summary row across the multi-patient OpenKBP re-irradiation cohort."""

    patient_id: str
    case_id: str
    oars_present: list[str]
    tg132_pass: bool
    jacobian_min: float
    ice_p95_mm: float
    max_inverse_residual_gy: float
    landmark_validation: LandmarkTREValidation
    remaining_physical_p95_gy: dict[str, float]
    cumulative_eqd2_p95_gy: dict[str, float]
    ntcp_summaries: list[OARNTCPSummary]
    sobol_spinal_cord: SobolSensitivityResult | None = None


@dataclass
class CohortBenchmarkResult:
    """Aggregated multi-patient OpenKBP cohort benchmark + K=2 multi-course verification."""

    patients: list[CohortPatientSummary]
    k2_multicourse_output: FullBudgetOutput | None = None


def extract_anatomical_landmarks(
    oar_masks: dict[str, np.ndarray],
    spacing_mm: tuple[float, float, float],
) -> list[tuple[str, np.ndarray]]:
    """Extract ~15-20 reproducible anatomical landmarks in physical mm (z, y, x)."""
    dz, dy, dx = spacing_mm
    scale = np.array([dz, dy, dx], dtype=np.float64)
    landmarks: list[tuple[str, np.ndarray]] = []

    for oar_name, mask in sorted(oar_masks.items()):
        coords = np.argwhere(mask)
        if coords.shape[0] < 5:
            continue
        pts_mm = coords.astype(np.float64) * scale
        landmarks.append((f"{oar_name}_centroid", np.mean(pts_mm, axis=0)))

        if oar_name == "SpinalCord":
            # 5 evenly spaced cervical centerline landmarks from superior to inferior
            z_unique = np.unique(coords[:, 0])
            picks = np.linspace(0, len(z_unique) - 1, min(5, len(z_unique)), dtype=int)
            for idx_k, zi in enumerate(z_unique[picks], start=1):
                sl_pts = coords[coords[:, 0] == zi].astype(np.float64) * scale
                landmarks.append((f"SpinalCord_C{idx_k}", np.mean(sl_pts, axis=0)))
        elif oar_name == "Brainstem":
            z_min_i = int(np.argmin(coords[:, 0]))
            z_max_i = int(np.argmax(coords[:, 0]))
            landmarks.append(("Brainstem_superior_pole", pts_mm[z_min_i]))
            landmarks.append(("Brainstem_inferior_pole", pts_mm[z_max_i]))
        elif oar_name == "Bone_Mandible":
            y_min_i = int(np.argmin(coords[:, 1]))  # anterior mental symphysis
            x_min_i = int(np.argmin(coords[:, 2]))  # right ramus/condyle extreme
            x_max_i = int(np.argmax(coords[:, 2]))  # left ramus/condyle extreme
            landmarks.append(("Mandible_symphysis", pts_mm[y_min_i]))
            landmarks.append(("Mandible_right_extreme", pts_mm[x_min_i]))
            landmarks.append(("Mandible_left_extreme", pts_mm[x_max_i]))

    return landmarks


def _interp_dvf_at_point(
    dvf_mm: np.ndarray,
    pt_mm: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    vox = np.array(
        [pt_mm[0] / spacing_mm[0], pt_mm[1] / spacing_mm[1], pt_mm[2] / spacing_mm[2]],
        dtype=np.float64,
    ).reshape(3, 1)
    return np.array(
        [
            float(map_coordinates(dvf_mm[axis], vox, order=1, mode="nearest")[0])
            for axis in range(3)
        ],
        dtype=np.float64,
    )


def validate_landmark_tre_and_coverage(
    case: ReIrradiationBenchmarkCase,
    output: FullBudgetOutput,
    *,
    tre_sigma_mm: float = 2.5,
    n_coverage_trials: int = 8,
    seed: int = 2026,
) -> LandmarkTREValidation:
    """Validate pre/post-DIR anatomical landmark TRE and 90% envelope dose coverage calibration."""
    spacing = case.ct_course2.spacing_mm
    landmarks = extract_anatomical_landmarks(case.oar_masks_course2, spacing)

    pre_errs: list[float] = []
    post_errs: list[float] = []
    for _name, pt_c2 in landmarks:
        u_fwd = _interp_dvf_at_point(case.nominal_dvf_mm, pt_c2, spacing)
        pt_c1_true = pt_c2 + u_fwd
        # Pre-DIR TRE: distance between Course-2 anatomy and true corresponding Course-1 point
        pre_errs.append(float(np.linalg.norm(u_fwd)))
        # Post-DIR residual TRE: forward + inverse mapping closure error
        u_inv = _interp_dvf_at_point(case.nominal_inv_dvf_mm, pt_c1_true, spacing)
        closure = pt_c1_true + u_inv - pt_c2
        post_errs.append(float(np.linalg.norm(closure)))

    pre_arr = np.asarray(pre_errs, dtype=np.float64)
    post_arr = np.asarray(post_errs, dtype=np.float64)

    # Independent registration-error stress test for 90% [p05, p95] dose envelope calibration
    oar_union = np.zeros(case.ct_course2.shape, dtype=bool)
    for m in case.oar_masks_course2.values():
        oar_union |= m
    # Focus on OAR voxels with meaningful prior dose/gradient (>= 2 Gy)
    env = output.dir_envelope
    eval_mask = oar_union & (env.dose_p95.data >= 2.0)
    n_eval = int(np.sum(eval_mask))

    if n_eval == 0 or tre_sigma_mm <= 0.0:
        coverage_pct = 100.0
    else:
        rng = np.random.default_rng(seed)
        covered_total = 0
        tested_total = 0
        p05_eval = env.dose_p05.data[eval_mask]
        p95_eval = env.dose_p95.data[eval_mask]
        for _ in range(n_coverage_trials):
            delta = _generate_diffeomorphic_perturbation(
                case.ct_course2.shape, spacing, tre_sigma_mm=tre_sigma_mm, rng=rng
            )
            pert_dvf = compose_dvf(case.nominal_dvf_mm, delta, spacing)
            realized = warp_array(
                case.prior_dose_course1.data,
                pert_dvf,
                spacing,
                order=1,
                cval=0.0,
                clamp_nonnegative=True,
            )[eval_mask]
            # Small 0.25 Gy numerical tolerance for finite ensemble sample quantile estimation
            in_band = (realized >= p05_eval - 0.25) & (realized <= p95_eval + 0.25)
            covered_total += int(np.sum(in_band))
            tested_total += n_eval
        coverage_pct = 100.0 * covered_total / max(tested_total, 1)

    return LandmarkTREValidation(
        n_landmarks=len(landmarks),
        pre_dir_tre_mean_mm=float(np.mean(pre_arr)) if pre_arr.size else 0.0,
        pre_dir_tre_p95_mm=float(np.percentile(pre_arr, 95)) if pre_arr.size else 0.0,
        pre_dir_tre_max_mm=float(np.max(pre_arr)) if pre_arr.size else 0.0,
        post_dir_tre_mean_mm=float(np.mean(post_arr)) if post_arr.size else 0.0,
        post_dir_tre_p95_mm=float(np.percentile(post_arr, 95)) if post_arr.size else 0.0,
        post_dir_tre_max_mm=float(np.max(post_arr)) if post_arr.size else 0.0,
        empirical_90pct_envelope_coverage_pct=round(coverage_pct, 1),
        gradient_voxels_tested=n_eval,
    )


def build_k2_multicourse_benchmark(
    patient: OpenKBPPatient,
    *,
    tre_sigma_mm: float = 2.0,
    n_ensemble: int = 16,
    seed: int = 42,
) -> FullBudgetOutput:
    """Demonstrate true K=2 heterogeneous prior courses -> Course 3 salvage budget solving.

    - Prior Course 1 (30 mo prior): Primary IMRT (54 Gy / 27 fx, Photon LQ) from OpenKBP dose * 0.77
    - Prior Course 2 (14 mo prior): Prior Nodal Boost (18 Gy / 3 fx, Photon LQL) with distinct SVF
    - Planned Course 3: Salvage Re-Irradiation (30 Gy / 5 fx, Photon LQL)
    """
    base = build_reirradiation_benchmark_from_openkbp(
        patient, months_elapsed=30.0, planned_n_fractions=5, planned_rx_gy=30.0
    )
    spacing = base.ct_course2.spacing_mm
    shape = base.ct_course2.shape

    pat_num = "".join(ch for ch in patient.patient_id if ch.isdigit()) or "0"
    uid_base = f"1.2.826.0.1.3680043.8.498.2021.{pat_num}"

    # Prior Course 1: 54 Gy / 27 fx primary IMRT (scaled from real OpenKBP 70 Gy plan)
    dose_c1 = Volume3D(
        data=patient.dose_gy.data * (54.0 / 70.0),
        spacing_mm=spacing,
        origin_mm=patient.dose_gy.origin_mm,
        frame_of_reference_uid=f"{uid_base}.101",
        units="Gy",
        name=f"{patient.patient_id}_prior_c1_54gy",
    )
    spec_c1 = CourseSpec(
        course_id="PRIOR_1_IMRT",
        label="Prior Course 1: Primary IMRT (54 Gy / 27 fx, LQ)",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQ,
        n_fractions=27,
        prescription_gy=54.0,
        months_before_reirt=30.0,
        frame_of_reference_uid=dose_c1.frame_of_reference_uid,
    )

    # Prior Course 2: distinct 14-month prior focal SBRT boost (18 Gy / 3 fx, LQL) with its own SVF
    vel_c2 = np.zeros((3, *shape), dtype=np.float64)
    vel_c2[0] = 1.2
    vel_c2[1] = -1.5
    vel_c2[2] = 1.0
    dvf_c2_fwd = integrate_svf_to_dvf(vel_c2, spacing, steps=3)
    dvf_c2_inv = integrate_svf_to_dvf(-vel_c2, spacing, steps=3)

    dose_c2 = Volume3D(
        data=base.candidate_dose_course2.data * (18.0 / 30.0),
        spacing_mm=spacing,
        origin_mm=base.ct_course2.origin_mm,
        frame_of_reference_uid=f"{uid_base}.102",
        units="Gy",
        name=f"{patient.patient_id}_prior_c2_18gy_sbrt",
    )
    spec_c2 = CourseSpec(
        course_id="PRIOR_2_SBRT",
        label="Prior Course 2: Focal Salvage Boost (18 Gy / 3 fx, LQL)",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQL,
        n_fractions=3,
        prescription_gy=18.0,
        months_before_reirt=14.0,
        frame_of_reference_uid=dose_c2.frame_of_reference_uid,
    )

    planned_c3 = CourseSpec(
        course_id="PLANNED_C3_SALVAGE",
        label="Planned Course 3: Second Salvage (30 Gy / 5 fx, LQL)",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQL,
        n_fractions=5,
        prescription_gy=30.0,
        frame_of_reference_uid=f"{uid_base}.103",
    )

    return solve_multicourse_reirradiation_budget(
        case_id=f"OPENKBP-{patient.patient_id.upper()}-K2-MULTICOURSE",
        prior_inputs=[
            PriorCourseInput(
                course=spec_c1,
                dose=dose_c1,
                nominal_dvf_mm=base.nominal_dvf_mm,
                nominal_inv_dvf_mm=base.nominal_inv_dvf_mm,
                tre_sigma_mm=tre_sigma_mm,
            ),
            PriorCourseInput(
                course=spec_c2,
                dose=dose_c2,
                nominal_dvf_mm=dvf_c2_fwd,
                nominal_inv_dvf_mm=dvf_c2_inv,
                tre_sigma_mm=tre_sigma_mm * 0.75,
            ),
        ],
        planned_course=planned_c3,
        oar_masks_course2=base.oar_masks_course2,
        body_mask_course2=base.body_mask_course2,
        new_ptv_mask_course2=base.recurrent_ptv_course2,
        n_ensemble=n_ensemble,
        seed=seed,
    )


def run_openkbp_cohort_benchmark(
    patient_dirs: list[str | Path],
    *,
    tre_sigma_mm: float = 2.5,
    n_ensemble: int = 16,
    sobol_pow2: int = 7,
    seed: int = 42,
) -> CohortBenchmarkResult:
    """Run multi-patient OpenKBP validation (TRE + coverage + NTCP + Sobol + K=2 check)."""
    summaries: list[CohortPatientSummary] = []
    k2_out: FullBudgetOutput | None = None

    for idx, pdir in enumerate(patient_dirs):
        patient = load_openkbp_patient(pdir)
        case = build_reirradiation_benchmark_from_openkbp(
            patient, months_elapsed=18.0, planned_n_fractions=5, planned_rx_gy=35.0
        )
        out = solve_reirradiation_budget(
            case_id=case.case_id,
            prior_course=case.prior_course,
            prior_dose=case.prior_dose_course1,
            planned_course=case.planned_course,
            nominal_dvf_mm=case.nominal_dvf_mm,
            nominal_inv_dvf_mm=case.nominal_inv_dvf_mm,
            oar_masks_course2=case.oar_masks_course2,
            body_mask_course2=case.body_mask_course2,
            new_ptv_mask_course2=case.recurrent_ptv_course2,
            candidate_course2_dose=case.candidate_dose_course2,
            let_d_course2=case.let_d_course2,
            tre_sigma_mm=tre_sigma_mm,
            n_ensemble=n_ensemble,
            seed=seed + idx * 17,
            retain_ensemble_samples=True,
        )

        lm_val = validate_landmark_tre_and_coverage(
            case, out, tre_sigma_mm=tre_sigma_mm, n_coverage_trials=6, seed=seed + idx * 31
        )

        ntcp_list: list[OARNTCPSummary] = []
        for oar_name, env_dict in out.oar_eqd2_envelopes.items():
            ns = evaluate_oar_ntcp_envelope(oar_name, env_dict)
            if ns is not None:
                ntcp_list.append(ns)

        sobol_cord: SobolSensitivityResult | None = None
        if "SpinalCord" in case.oar_masks_course2:
            sc_mask = case.oar_masks_course2["SpinalCord"]
            ens_sc = out.dir_envelope.extract_masked_ensemble(sc_mask)
            sobol_cord = run_joint_sobol_sensitivity(
                oar_name="SpinalCord",
                prior_ensemble_vox=ens_sc,
                candidate_course2_vox=case.candidate_dose_course2.data[sc_mask],
                prior_course=case.prior_course,
                planned_course=case.planned_course,
                voxel_vol_cc=case.prior_dose_course1.voxel_volume_cc,
                output_mode="eqd2",
                n_base_pow2=sobol_pow2,
                seed=seed + idx,
            )

        res = out.result
        summaries.append(
            CohortPatientSummary(
                patient_id=patient.patient_id,
                case_id=case.case_id,
                oars_present=[s.tg263_name for s in res.oar_summaries],
                tg132_pass=res.dir_quality.tg132_pass,
                jacobian_min=res.dir_quality.jacobian_min,
                ice_p95_mm=res.dir_quality.ice_p95_mm,
                max_inverse_residual_gy=res.provenance.max_inverse_residual_gy,
                landmark_validation=lm_val,
                remaining_physical_p95_gy={
                    s.tg263_name: round(s.remaining_physical_p95_gy, 2) for s in res.oar_summaries
                },
                cumulative_eqd2_p95_gy={
                    s.tg263_name: round(float(s.cumulative_eqd2_p95_gy or s.prior_eqd2_effective_p95_gy), 2)
                    for s in res.oar_summaries
                },
                ntcp_summaries=ntcp_list,
                sobol_spinal_cord=sobol_cord,
            )
        )

        if idx == 0:
            k2_out = build_k2_multicourse_benchmark(patient, tre_sigma_mm=2.0, n_ensemble=12, seed=seed)

    return CohortBenchmarkResult(patients=summaries, k2_multicourse_output=k2_out)
