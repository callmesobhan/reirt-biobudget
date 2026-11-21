"""Lyman-Kutcher-Burman (LKB) / gEUD Cumulative NTCP & Saltelli-Sobol Global Sensitivity.

Implements:
1. Numerically overflow-safe Generalized Equivalent Uniform Dose (gEUD) and LKB Normal
   Tissue Complication Probability (NTCP) with 90% uncertainty confidence intervals
   propagated through the 3D DIR-uncertainty envelope.
2. Exact quasi-Monte Carlo Saltelli-Jansen global Sobol sensitivity decomposition
   (First-order S_i, Total-order S_Ti, and Interaction S_Ti - S_i) across the 4D joint
   uncertainty space: DIR spatial registration x Tissue recovery x alpha/beta x High-dose regime.

Peer-Reviewed References:
    - Lyman JT. Complication probability as assessed from dose-volume histograms.
      Radiat Res Suppl. 1985;8:S13-S19. DOI: 10.2307/3583506
    - Burman C, Kutcher GJ, Emami B, Goitein M. Fitting of normal tissue tolerance data
      to an analytic function. Int J Radiat Oncol Biol Phys. 1991;21(1):123-135.
      DOI: 10.1016/0360-3016(91)90172-Z
    - Kirkpatrick JP et al. QUANTEC: Radiation dose-volume effects in the spinal cord.
      Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S42-S49.
      DOI: 10.1016/j.ijrobp.2009.04.095
    - Deasy JO et al. QUANTEC: Radiotherapy dose-volume effects on salivary gland function.
      Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S58-S63.
      DOI: 10.1016/j.ijrobp.2009.06.090
    - Saltelli A et al. Variance based sensitivity analysis of model output. Design and
      estimator for the total sensitivity index. Comput Phys Commun. 2010;181(2):259-270.
      DOI: 10.1016/j.cpc.2009.09.018
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf
from scipy.stats import qmc

from reirt_biobudget.budget_solver import evaluate_oar_metric
from reirt_biobudget.ir import (
    CourseSpec,
    LKBModelSpec,
    OARNTCPSummary,
    RadiobiologyModel,
    SobolSensitivityResult,
)
from reirt_biobudget.ontology import get_oar_constraint, normalize_tg263
from reirt_biobudget.radiobiology import physical_to_eqd2

# Sourced LKB NTCP parameter library mapped to AAPM TG-263 OAR names
LKB_OAR_LIBRARY: dict[str, LKBModelSpec] = {
    "SpinalCord": LKBModelSpec(
        tg263_name="SpinalCord",
        endpoint="Radiation myelopathy (Grade >= 3)",
        td50_gy=66.5,
        m=0.175,
        n=0.05,
        citation=(
            "Burman et al. (IJROBP 1991;21:123-135); "
            "Kirkpatrick et al. QUANTEC (IJROBP 2010;76:S42-S49)"
        ),
        doi="10.1016/j.ijrobp.2009.04.095",
    ),
    "Brainstem": LKBModelSpec(
        tg263_name="Brainstem",
        endpoint="Brainstem necrosis / neuropathy",
        td50_gy=65.0,
        m=0.14,
        n=0.16,
        citation="Burman et al. (IJROBP 1991;21:123-135); Marks et al. QUANTEC (IJROBP 2010)",
        doi="10.1016/0360-3016(91)90172-Z",
    ),
    "Bone_Mandible": LKBModelSpec(
        tg263_name="Bone_Mandible",
        endpoint="Osteoradionecrosis (ORN)",
        td50_gy=72.0,
        m=0.10,
        n=0.07,
        citation="Burman et al. (IJROBP 1991;21:123-135)",
        doi="10.1016/0360-3016(91)90172-Z",
    ),
    "Parotid_L": LKBModelSpec(
        tg263_name="Parotid_L",
        endpoint="Severe xerostomia (<25% salivary flow)",
        td50_gy=39.9,
        m=0.40,
        n=1.00,
        citation="Deasy et al. QUANTEC (IJROBP 2010;76:S58-S63)",
        doi="10.1016/j.ijrobp.2009.06.090",
    ),
    "Parotid_R": LKBModelSpec(
        tg263_name="Parotid_R",
        endpoint="Severe xerostomia (<25% salivary flow)",
        td50_gy=39.9,
        m=0.40,
        n=1.00,
        citation="Deasy et al. QUANTEC (IJROBP 2010;76:S58-S63)",
        doi="10.1016/j.ijrobp.2009.06.090",
    ),
    "Larynx": LKBModelSpec(
        tg263_name="Larynx",
        endpoint="Symptomatic laryngeal edema",
        td50_gy=70.0,
        m=0.15,
        n=0.45,
        citation="Burman et al. (IJROBP 1991;21:123-135); Marks et al. QUANTEC (IJROBP 2010)",
        doi="10.1016/j.ijrobp.2009.07.1754",
    ),
    "Esophagus": LKBModelSpec(
        tg263_name="Esophagus",
        endpoint="Esophageal stricture / perforation",
        td50_gy=68.0,
        m=0.11,
        n=0.06,
        citation="Burman et al. (IJROBP 1991;21:123-135)",
        doi="10.1016/0360-3016(91)90172-Z",
    ),
}


def get_lkb_spec(raw_name: str) -> LKBModelSpec | None:
    """Return sourced LKB NTCP specification for a raw or canonical TG-263 OAR name."""
    return LKB_OAR_LIBRARY.get(normalize_tg263(raw_name))


def compute_geud(dose_values_gy: np.ndarray, n: float) -> float:
    """Overflow-safe Generalized Equivalent Uniform Dose: gEUD = (mean(D^(1/n)))^n."""
    if n <= 0.0:
        raise ValueError("LKB volume parameter n must be > 0")
    d = np.maximum(np.asarray(dose_values_gy, dtype=np.float64), 0.0)
    if d.size == 0:
        return 0.0
    d_max = float(np.max(d))
    if d_max <= 1e-12:
        return 0.0
    if abs(n - 1.0) < 1e-12:
        return float(np.mean(d))
    # Factor out d_max to prevent float64 overflow when 1/n is large (e.g., n = 0.05 -> 1/n = 20)
    inv_n = 1.0 / float(n)
    scaled = np.power(d / d_max, inv_n)
    return float(d_max * (float(np.mean(scaled)) ** float(n)))


def compute_lkb_ntcp_from_geud(geud_gy: float | np.ndarray, td50_gy: float, m: float) -> np.ndarray:
    """Evaluate LKB NTCP probability in [0, 1] from gEUD (Gy), TD50 (Gy), and slope m."""
    if td50_gy <= 0.0 or m <= 0.0:
        raise ValueError("TD50 and m must be positive")
    t = (np.asarray(geud_gy, dtype=np.float64) - float(td50_gy)) / (float(m) * float(td50_gy))
    return 0.5 * (1.0 + erf(t / math.sqrt(2.0)))


def evaluate_oar_ntcp_envelope(
    oar_name: str,
    oar_envelope: dict[str, np.ndarray],
    spec: LKBModelSpec | None = None,
) -> OARNTCPSummary | None:
    """Evaluate cumulative gEUD and LKB NTCP (%) with 90% confidence band for one OAR."""
    lkb = spec or get_lkb_spec(oar_name)
    if lkb is None:
        return None
    d_p05 = oar_envelope["cum_p05"]
    d_nom = oar_envelope["cum_nominal"]
    d_p95 = oar_envelope["cum_p95"]
    d_worst = oar_envelope["cum_worst"]

    geud_p05 = compute_geud(d_p05, lkb.n)
    geud_nom = compute_geud(d_nom, lkb.n)
    geud_p95 = compute_geud(d_p95, lkb.n)
    geud_worst = compute_geud(d_worst, lkb.n)

    ntcp_p05 = float(compute_lkb_ntcp_from_geud(geud_p05, lkb.td50_gy, lkb.m)) * 100.0
    ntcp_nom = float(compute_lkb_ntcp_from_geud(geud_nom, lkb.td50_gy, lkb.m)) * 100.0
    ntcp_p95 = float(compute_lkb_ntcp_from_geud(geud_p95, lkb.td50_gy, lkb.m)) * 100.0
    ntcp_worst = float(compute_lkb_ntcp_from_geud(geud_worst, lkb.td50_gy, lkb.m)) * 100.0

    return OARNTCPSummary(
        tg263_name=lkb.tg263_name,
        endpoint=lkb.endpoint,
        geud_p05_gy=geud_p05,
        geud_nominal_gy=geud_nom,
        geud_p95_gy=geud_p95,
        ntcp_p05_pct=ntcp_p05,
        ntcp_nominal_pct=ntcp_nom,
        ntcp_p95_pct=ntcp_p95,
        ntcp_worst_pct=ntcp_worst,
        td50_gy=lkb.td50_gy,
        m=lkb.m,
        n=lkb.n,
        citation=lkb.citation,
        doi=lkb.doi,
    )


def run_joint_sobol_sensitivity(
    *,
    oar_name: str,
    prior_ensemble_vox: np.ndarray,  # shape (M, N_vox) warped Course-1 physical dose samples
    candidate_course2_vox: np.ndarray,  # shape (N_vox,) Course-2 physical dose
    prior_course: CourseSpec,
    planned_course: CourseSpec,
    voxel_vol_cc: float,
    output_mode: str = "eqd2",  # "eqd2" or "ntcp"
    n_base_pow2: int = 8,  # N = 2^8 = 256 -> 256 * 6 = 1536 QMC evaluations
    seed: int = 42,
) -> SobolSensitivityResult:
    """Compute exact Saltelli-Jansen first-order (S_i) and total-order (S_Ti) Sobol indices.

    Jointly propagates 4 physical/biological uncertainty dimensions:
      1. DIR Spatial Registration (stratified across 3D diffeomorphic ensemble draws)
      2. Tissue Recovery Credit r ~ U(0, max_recovery_fraction)
      3. Radiobiology alpha/beta ~ U(alpha_beta_low, alpha_beta_high)
      4. High-Dose / Modality Regime (LQL d_T in [4.5, 9.5] Gy or Course-2 dose calibration +/- 5%)
    """
    tg263 = normalize_tg263(oar_name)
    cons = get_oar_constraint(tg263)
    if cons is None:
        raise ValueError(f"No TG-263 OAR constraint found for '{oar_name}'")
    lkb = get_lkb_spec(tg263)
    if output_mode == "ntcp" and lkb is None:
        raise ValueError(f"No LKB NTCP specification found for '{oar_name}'")

    ens = np.asarray(prior_ensemble_vox, dtype=np.float64)
    if ens.ndim == 1:
        ens = ens[np.newaxis, :]
    m_ens = ens.shape[0]
    c2_vox = np.asarray(candidate_course2_vox, dtype=np.float64)

    # Sort ensemble members monotonically by OAR primary metric so u_dir in [0, 1) maps smoothly
    ens_scores = np.array(
        [evaluate_oar_metric(ens[i], cons.primary_metric, voxel_vol_cc) for i in range(m_ens)]
    )
    order = np.argsort(ens_scores)
    ens_sorted = ens[order]

    param_names = [
        "DIR Spatial Registration",
        "Tissue Recovery Credit",
        "Radiobiology alpha/beta",
        "Course-2 High-Dose Regime",
    ]
    k = len(param_names)
    n_base = 2**n_base_pow2

    # Generate independent Sobol QMC matrices A and B in [0, 1)^k
    sampler = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    u_full = sampler.random_base2(m=n_base_pow2)
    mat_a = u_full[:, :k]
    mat_b = u_full[:, k:]

    rec_max = max(cons.max_recovery_fraction, 0.05)
    ab_lo = cons.alpha_beta_low_gy
    ab_hi = cons.alpha_beta_high_gy
    use_lql = planned_course.model == RadiobiologyModel.LQL

    def _eval_batch(u_mat: np.ndarray) -> np.ndarray:
        n_rows = u_mat.shape[0]
        y_out = np.empty(n_rows, dtype=np.float64)
        for i in range(n_rows):
            u_dir, u_rec, u_ab, u_reg = u_mat[i]
            # Interpolate smoothly between adjacent sorted 3D DIR ensemble draws
            pos = u_dir * max(m_ens - 1, 0)
            i0 = math.floor(pos)
            i1 = min(i0 + 1, m_ens - 1)
            w = pos - i0
            d1_vox = (1.0 - w) * ens_sorted[i0] + w * ens_sorted[i1]

            rec_frac = u_rec * rec_max
            ab = ab_lo + u_ab * (ab_hi - ab_lo)

            e1 = physical_to_eqd2(
                d1_vox,
                prior_course,
                ab,
                repair_half_time_h=cons.repair_half_time_h,
                lql_d_t_gy=cons.lql_d_t_gy,
            ) * (1.0 - rec_frac)

            if use_lql:
                d_t_sample = 4.5 + u_reg * 5.0  # LQL transition dose d_T in [4.5, 9.5] Gy
                d2_scaled = c2_vox * (0.97 + 0.06 * u_reg)  # includes +-3% delivery calibration
                e2 = physical_to_eqd2(
                    d2_scaled,
                    planned_course,
                    ab,
                    repair_half_time_h=cons.repair_half_time_h,
                    lql_d_t_gy=d_t_sample,
                )
            else:
                d2_scaled = c2_vox * (0.95 + 0.10 * u_reg)  # +-5% delivery/LET calibration
                e2 = physical_to_eqd2(
                    d2_scaled,
                    planned_course,
                    ab,
                    repair_half_time_h=cons.repair_half_time_h,
                )

            cum_vox = e1 + e2
            if output_mode == "ntcp":
                assert lkb is not None
                geud = compute_geud(cum_vox, lkb.n)
                y_out[i] = float(compute_lkb_ntcp_from_geud(geud, lkb.td50_gy, lkb.m)) * 100.0
            else:
                y_out[i] = evaluate_oar_metric(cum_vox, cons.primary_metric, voxel_vol_cc)
        return y_out

    y_a = _eval_batch(mat_a)
    y_b = _eval_batch(mat_b)
    y_all = np.concatenate([y_a, y_b])
    var_y = float(np.var(y_all, ddof=1))

    s1_map: dict[str, float] = {}
    st_map: dict[str, float] = {}
    int_map: dict[str, float] = {}

    for col_i, pname in enumerate(param_names):
        mat_ab_i = mat_a.copy()
        mat_ab_i[:, col_i] = mat_b[:, col_i]
        y_ab_i = _eval_batch(mat_ab_i)

        if var_y < 1e-12:
            s1_val = 0.0
            st_val = 0.0
        else:
            # Saltelli (2010) Eq. (b) for S_i and Jansen (1999) / Saltelli (2010) Eq. (f) for S_Ti
            s1_val = float(np.mean(y_b * (y_ab_i - y_a)) / var_y)
            st_val = float(0.5 * np.mean((y_a - y_ab_i) ** 2) / var_y)
            s1_val = float(np.clip(s1_val, 0.0, 1.0))
            st_val = float(np.clip(st_val, 0.0, 1.0))

        s1_map[pname] = round(s1_val, 4)
        st_map[pname] = round(st_val, 4)
        int_map[pname] = round(max(st_val - s1_val, 0.0), 4)

    dominant = max(st_map, key=lambda name: st_map[name])
    metric_label = (
        f"LKB NTCP (%) [{lkb.endpoint}]" if (output_mode == "ntcp" and lkb) else f"Cum. EQD2 {cons.primary_metric} (Gy)"
    )

    return SobolSensitivityResult(
        oar_name=tg263,
        output_metric=metric_label,
        n_base_samples=n_base,
        total_evaluations=n_base * (k + 2),
        parameter_names=param_names,
        first_order_s1=s1_map,
        total_order_st=st_map,
        interaction_st_minus_s1=int_map,
        output_mean=float(np.mean(y_all)),
        output_std=float(np.std(y_all, ddof=1)),
        output_p05=float(np.percentile(y_all, 5)),
        output_p50=float(np.percentile(y_all, 50)),
        output_p95=float(np.percentile(y_all, 95)),
        dominant_parameter=dominant,
    )
