"""Proprietary Multi-Regime 3D Radiobiology Forward & Exact Inverse Engine.

Implements vectorized, machine-precision forward (physical dose -> BED / EQD2) and
exact analytical inverse (remaining EQD2 -> physical dose budget) transformations
across four peer-reviewed radiobiological regimes:

1. Standard Linear-Quadratic (LQ) — Fowler (1989), BJR 62:679-694.
   DOI: 10.1259/0007-1285-62-740-679
2. Linear-Quadratic-Linear (LQL, C1-continuous) — Astrahan (2008), Med Phys 35:4161.
   DOI: 10.1118/1.2969065
3. Lea-Catcheside Incomplete Repair (protracted / HDR / BID) — Dale (1985), BJR 58:515.
   DOI: 10.1259/0007-1285-58-690-515
4. Phenomenological Variable Proton RBE from LET_d — McNamara et al. (2015), PMB 60:8399.
   DOI: 10.1088/0031-9155/60/21/8399

Time-dependent OAR recovery follows conservative clinical ceilings from:
- Nieder et al. (2005), IJROBP 61:851-855. DOI: 10.1016/j.ijrobp.2004.06.016
- Nieder et al. (2006), IJROBP 66:1446-1449. DOI: 10.1016/j.ijrobp.2006.07.1383
- Sahgal et al. (2021), HyTEC IJROBP 110:124-136. DOI: 10.1016/j.ijrobp.2019.09.038
- Andratschke et al. (2022), ESTRO/EORTC Lancet Oncol 23:e469-e478.
"""

from __future__ import annotations

import math

import numpy as np

from reirt_biobudget.ir import CourseSpec, Modality, OARConstraint, RadiobiologyModel

# McNamara et al. (2015) phenomenological proton RBE coefficients (PMB 60:8399, Table 2)
MCNAMARA_P0 = 0.99064
MCNAMARA_P1 = 0.35605  # Gy * um / keV
MCNAMARA_P2 = 1.1012
MCNAMARA_P3 = -0.0038703  # Gy^(-1/2) * um / keV


def lea_catcheside_g_continuous(duration_h: float, repair_half_time_h: float) -> float:
    """Compute continuous-exposure Lea-Catcheside factor G(mu*T) in (0, 1].

    Uses a 4th-order Taylor expansion for mu*T < 1e-3 to eliminate catastrophic
    cancellation while maintaining < 1e-13 accuracy across the entire domain.
    """
    if duration_h <= 0.0:
        return 1.0
    if repair_half_time_h <= 0.0:
        raise ValueError("repair_half_time_h must be positive")
    mu = math.log(2.0) / repair_half_time_h
    x = mu * duration_h
    if x < 1e-3:
        return 1.0 - x / 3.0 + (x * x) / 12.0 - (x * x * x) / 60.0
    return float((2.0 / (x * x)) * (x - 1.0 + math.exp(-x)))


def incomplete_repair_hm(
    fractions_per_day: int,
    interfraction_interval_h: float,
    repair_half_time_h: float,
) -> float:
    """Compute inter-fraction incomplete repair factor h_m (Thames / Dale 1985).

    For once-daily schedules (fractions_per_day == 1), returns 0.0.
    For multi-fraction/day schedules (e.g., BID at 6h intervals), computes the exact
    geometric sum for m equally spaced fractions.
    """
    if fractions_per_day <= 1:
        return 0.0
    if interfraction_interval_h <= 0.0 or repair_half_time_h <= 0.0:
        raise ValueError("Intervals and repair half-time must be positive")
    mu = math.log(2.0) / repair_half_time_h
    theta = math.exp(-mu * interfraction_interval_h)
    m = float(fractions_per_day)
    if theta >= 1.0 - 1e-12:
        return float(m - 1.0)
    return float((2.0 / m) * (theta / (1.0 - theta)) * (m - (1.0 - theta**m) / (1.0 - theta)))


def mcnamara_rbe_bounds(
    alpha_beta_gy: float,
    let_d_kev_um: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute McNamara et al. (2015) RBE_max and RBE_min from alpha/beta and LET_d."""
    if alpha_beta_gy <= 0.0:
        raise ValueError("alpha_beta_gy must be positive")
    let_arr = np.maximum(np.asarray(let_d_kev_um, dtype=np.float64), 0.0)
    rbe_max = MCNAMARA_P0 + (MCNAMARA_P1 / alpha_beta_gy) * let_arr
    rbe_min = np.maximum(MCNAMARA_P2 + MCNAMARA_P3 * math.sqrt(alpha_beta_gy) * let_arr, 0.5)
    return rbe_max, rbe_min


def mcnamara_rbe(
    dose_per_fraction_gy: np.ndarray | float,
    alpha_beta_gy: float,
    let_d_kev_um: np.ndarray | float,
) -> np.ndarray:
    """Evaluate voxel-wise McNamara et al. (2015) proton RBE(d_p, alpha/beta, LET_d)."""
    d_p = np.maximum(np.asarray(dose_per_fraction_gy, dtype=np.float64), 0.0)
    rbe_max, rbe_min = mcnamara_rbe_bounds(alpha_beta_gy, let_d_kev_um)
    ab = float(alpha_beta_gy)
    disc = ab * ab + 4.0 * d_p * ab * rbe_max + 4.0 * (d_p * d_p) * (rbe_min * rbe_min)
    # Use limit RBE_max where d_p == 0 to avoid 0/0
    d_eff = 0.5 * (np.sqrt(disc) - ab)
    return np.where(d_p > 1e-12, d_eff / np.maximum(d_p, 1e-12), rbe_max)


def physical_to_bed(
    physical_dose_gy: np.ndarray | float,
    course: CourseSpec,
    alpha_beta_gy: float,
    *,
    repair_half_time_h: float = 1.5,
    lql_d_t_gy: float | None = None,
    let_d_kev_um: np.ndarray | float | None = None,
) -> np.ndarray:
    """Convert total physical dose D (Gy) to Biologically Effective Dose (BED, Gy)."""
    if alpha_beta_gy <= 0.0:
        raise ValueError("alpha_beta_gy must be > 0")
    dose = np.asarray(physical_dose_gy, dtype=np.float64)
    if np.any(dose < -1e-9):
        raise ValueError("physical_dose_gy cannot contain negative doses")
    dose = np.maximum(dose, 0.0)
    n = float(course.n_fractions)
    ab = float(alpha_beta_gy)

    # Apply fixed RBE scaling for standard clinical proton therapy (1.1x)
    if course.modality == Modality.PROTON_FIXED_RBE and course.model != RadiobiologyModel.MCNAMARA_PROTON:
        dose = dose * float(course.fixed_rbe)

    d = dose / n

    if course.model == RadiobiologyModel.LQ:
        return dose * (1.0 + d / ab)

    if course.model == RadiobiologyModel.LQL:
        d_t = float(course.d_t_gy or lql_d_t_gy or 6.0)
        if d_t <= 0.0:
            raise ValueError("LQL transition dose d_t must be > 0")
        bed_lq = dose * (1.0 + d / ab)
        bed_lin = n * (d_t * (1.0 + d_t / ab) + (d - d_t) * (1.0 + (2.0 * d_t) / ab))
        return np.where(d <= d_t, bed_lq, bed_lin)

    if course.model == RadiobiologyModel.LEA_CATCHESIDE:
        g_intra = lea_catcheside_g_continuous(course.fraction_duration_h, repair_half_time_h)
        h_inter = incomplete_repair_hm(
            course.fractions_per_day,
            course.interfraction_interval_h,
            repair_half_time_h,
        )
        g_eff = g_intra + h_inter
        return dose * (1.0 + (g_eff * d) / ab)

    if course.model == RadiobiologyModel.MCNAMARA_PROTON:
        let_field = course.mean_let_d_kev_um if let_d_kev_um is None else let_d_kev_um
        rbe_max, rbe_min = mcnamara_rbe_bounds(ab, let_field)
        # Exact identity from McNamara et al. (2015):
        # d_eff * (1 + d_eff / ab) == d_p * RBE_max + (d_p^2 * RBE_min^2) / ab
        return n * (d * rbe_max + (d * d) * (rbe_min * rbe_min) / ab)

    raise ValueError(f"Unsupported RadiobiologyModel: {course.model}")


def bed_to_eqd2(bed_gy: np.ndarray | float, alpha_beta_gy: float) -> np.ndarray:
    """Convert BED (Gy) to Equivalent Dose in 2 Gy fractions (EQD2, Gy)."""
    if alpha_beta_gy <= 0.0:
        raise ValueError("alpha_beta_gy must be > 0")
    return np.asarray(bed_gy, dtype=np.float64) / (1.0 + 2.0 / float(alpha_beta_gy))


def eqd2_to_bed(eqd2_gy: np.ndarray | float, alpha_beta_gy: float) -> np.ndarray:
    """Convert EQD2 (Gy) to BED (Gy)."""
    if alpha_beta_gy <= 0.0:
        raise ValueError("alpha_beta_gy must be > 0")
    return np.asarray(eqd2_gy, dtype=np.float64) * (1.0 + 2.0 / float(alpha_beta_gy))


def physical_to_eqd2(
    physical_dose_gy: np.ndarray | float,
    course: CourseSpec,
    alpha_beta_gy: float,
    *,
    repair_half_time_h: float = 1.5,
    lql_d_t_gy: float | None = None,
    let_d_kev_um: np.ndarray | float | None = None,
) -> np.ndarray:
    """Convert total physical dose D (Gy) directly to EQD2 (Gy)."""
    bed = physical_to_bed(
        physical_dose_gy,
        course,
        alpha_beta_gy,
        repair_half_time_h=repair_half_time_h,
        lql_d_t_gy=lql_d_t_gy,
        let_d_kev_um=let_d_kev_um,
    )
    return bed_to_eqd2(bed, alpha_beta_gy)


def eqd2_to_physical(
    target_eqd2_gy: np.ndarray | float,
    course: CourseSpec,
    alpha_beta_gy: float,
    *,
    repair_half_time_h: float = 1.5,
    lql_d_t_gy: float | None = None,
    let_d_kev_um: np.ndarray | float | None = None,
) -> np.ndarray:
    """Exact analytical inverse: convert remaining EQD2 (Gy) -> physical dose ceiling (Gy).

    Guarantees forward-inverse consistency residual < 1e-10 Gy across all four models.
    Negative target EQD2 values (already exceeded prior budget) clamp cleanly to 0.0 Gy.
    """
    if alpha_beta_gy <= 0.0:
        raise ValueError("alpha_beta_gy must be > 0")
    eqd2 = np.maximum(np.asarray(target_eqd2_gy, dtype=np.float64), 0.0)
    bed = eqd2_to_bed(eqd2, alpha_beta_gy)
    n = float(course.n_fractions)
    ab = float(alpha_beta_gy)

    if course.model == RadiobiologyModel.LQ:
        # Solve (1 / (n * ab)) * D^2 + D - BED = 0 using Citardauq form for stability near 0
        k = 1.0 / (n * ab)
        dose = (2.0 * bed) / (1.0 + np.sqrt(1.0 + 4.0 * k * bed))
        if course.modality == Modality.PROTON_FIXED_RBE:
            dose = dose / float(course.fixed_rbe)
        return dose

    if course.model == RadiobiologyModel.LQL:
        d_t = float(course.d_t_gy or lql_d_t_gy or 6.0)
        if d_t <= 0.0:
            raise ValueError("LQL transition dose d_t must be > 0")
        bed_t = n * d_t * (1.0 + d_t / ab)
        k = 1.0 / (n * ab)
        dose_lq = (2.0 * bed) / (1.0 + np.sqrt(1.0 + 4.0 * k * bed))
        slope = 1.0 + (2.0 * d_t) / ab
        d_lin = d_t + (bed / n - d_t * (1.0 + d_t / ab)) / slope
        dose = np.where(bed <= bed_t, dose_lq, n * d_lin)
        if course.modality == Modality.PROTON_FIXED_RBE:
            dose = dose / float(course.fixed_rbe)
        return dose

    if course.model == RadiobiologyModel.LEA_CATCHESIDE:
        g_intra = lea_catcheside_g_continuous(course.fraction_duration_h, repair_half_time_h)
        h_inter = incomplete_repair_hm(
            course.fractions_per_day,
            course.interfraction_interval_h,
            repair_half_time_h,
        )
        g_eff = g_intra + h_inter
        k = g_eff / (n * ab)
        dose = (2.0 * bed) / (1.0 + np.sqrt(1.0 + 4.0 * k * bed))
        if course.modality == Modality.PROTON_FIXED_RBE:
            dose = dose / float(course.fixed_rbe)
        return dose

    if course.model == RadiobiologyModel.MCNAMARA_PROTON:
        let_field = course.mean_let_d_kev_um if let_d_kev_um is None else let_d_kev_um
        rbe_max, rbe_min = mcnamara_rbe_bounds(ab, let_field)
        bed_per_fx = bed / n
        a_coeff = (rbe_min * rbe_min) / ab
        # Exact Citardauq root of a_coeff * d_p^2 + rbe_max * d_p - bed_per_fx = 0
        d_p = (2.0 * bed_per_fx) / (
            rbe_max + np.sqrt(rbe_max * rbe_max + 4.0 * a_coeff * bed_per_fx)
        )
        return n * d_p

    raise ValueError(f"Unsupported RadiobiologyModel: {course.model}")


def conservative_recovery_fraction(
    constraint: OARConstraint,
    months_elapsed: float,
    override_fraction: float | None = None,
) -> float:
    """Compute conservative OAR time-recovery discount factor r in [0, max_recovery_fraction].

    Strictly enforces zero recovery before `min_recovery_months` (e.g., 6 months for
    spinal cord/brainstem per Nieder et al. 2006 / Sahgal HyTEC 2021) and caps recovery
    at `max_recovery_fraction` (saturating linearly from min_recovery_months to 24 months).
    """
    if months_elapsed < 0.0:
        raise ValueError("months_elapsed cannot be negative")
    if override_fraction is not None:
        if not (0.0 <= override_fraction <= 0.5):
            raise ValueError(
                f"override_fraction={override_fraction} outside conservative physical bounds [0, 0.5]"
            )
        if months_elapsed < constraint.min_recovery_months and override_fraction > 0.0:
            raise ValueError(
                f"Cannot apply non-zero recovery ({override_fraction:.2f}) to {constraint.tg263_name} "
                f"when months_elapsed ({months_elapsed:.1f}) < min_recovery_months "
                f"({constraint.min_recovery_months:.1f})."
            )
        return float(override_fraction)

    if months_elapsed < constraint.min_recovery_months or constraint.max_recovery_fraction <= 0.0:
        return 0.0
    full_window = max(24.0 - constraint.min_recovery_months, 1.0)
    frac = min((months_elapsed - constraint.min_recovery_months) / full_window, 1.0)
    return float(frac * constraint.max_recovery_fraction)
