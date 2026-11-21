"""Rigorous mathematical and physical tests for the multi-regime radiobiology kernel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from reirt_biobudget.ir import CourseSpec, Modality, RadiobiologyModel
from reirt_biobudget.ontology import get_oar_constraint
from reirt_biobudget.radiobiology import (
    conservative_recovery_fraction,
    eqd2_to_physical,
    incomplete_repair_hm,
    lea_catcheside_g_continuous,
    mcnamara_rbe,
    physical_to_bed,
    physical_to_eqd2,
)

UID = "1.2.826.0.1.3680043.8.498.TEST.RADIOBIO"


def test_lq_closed_form_identities() -> None:
    # 70 Gy in 35 fx (2 Gy/fx) must equal 70.0 Gy EQD2 for any alpha/beta
    c_conv = CourseSpec(
        course_id="C1",
        label="70/35",
        n_fractions=35,
        prescription_gy=70.0,
        frame_of_reference_uid=UID,
    )
    for ab in (1.5, 2.0, 3.0, 10.0):
        eqd2 = float(physical_to_eqd2(70.0, c_conv, ab))
        assert math.isclose(eqd2, 70.0, rel_tol=0.0, abs_tol=1e-12)

    # 30 Gy in 5 fx (6 Gy/fx) with alpha/beta = 2 Gy -> BED = 30*(1+6/2) = 120, EQD2 = 120/2 = 60
    c_sbrt = CourseSpec(
        course_id="C2",
        label="30/5",
        n_fractions=5,
        prescription_gy=30.0,
        frame_of_reference_uid=UID,
    )
    assert math.isclose(float(physical_to_bed(30.0, c_sbrt, 2.0)), 120.0, abs_tol=1e-12)
    assert math.isclose(float(physical_to_eqd2(30.0, c_sbrt, 2.0)), 60.0, abs_tol=1e-12)


def test_lql_c1_continuity_and_high_dose_moderation() -> None:
    d_t = 6.0
    ab = 2.0
    c_lql = CourseSpec(
        course_id="LQL",
        label="LQL 5fx",
        model=RadiobiologyModel.LQL,
        n_fractions=5,
        prescription_gy=35.0,
        d_t_gy=d_t,
        frame_of_reference_uid=UID,
    )
    c_lq = c_lql.model_copy(update={"model": RadiobiologyModel.LQ})

    # Below d_T (d = 4 Gy -> D = 20 Gy): LQL must identically match LQ
    assert math.isclose(
        float(physical_to_bed(20.0, c_lql, ab)),
        float(physical_to_bed(20.0, c_lq, ab)),
        abs_tol=1e-12,
    )

    # C1 continuity at d = d_T (D = 30 Gy): left and right finite-difference derivatives match
    eps = 1e-6
    b_minus = float(physical_to_bed(30.0 - eps, c_lql, ab))
    b_zero = float(physical_to_bed(30.0, c_lql, ab))
    b_plus = float(physical_to_bed(30.0 + eps, c_lql, ab))
    deriv_left = (b_zero - b_minus) / eps
    deriv_right = (b_plus - b_zero) / eps
    assert math.isclose(deriv_left, deriv_right, rel_tol=1e-5)

    # Above d_T (d = 10 Gy -> D = 50 Gy): LQL moderates overprediction vs pure LQ
    assert float(physical_to_bed(50.0, c_lql, ab)) < float(physical_to_bed(50.0, c_lq, ab))


def test_lea_catcheside_limits_and_bid_repair() -> None:
    assert math.isclose(lea_catcheside_g_continuous(0.0, 1.5), 1.0, abs_tol=1e-15)
    # Taylor expansion continuity at mu*T ~ 1e-3
    g_tiny = lea_catcheside_g_continuous(1e-4, 1.5)
    assert 0.999 < g_tiny < 1.0
    # Longer protraction reduces G monotonically
    assert lea_catcheside_g_continuous(2.0, 1.5) < lea_catcheside_g_continuous(0.5, 1.5)
    # BID (6h interval) has positive incomplete repair factor h_2 > 0
    h2 = incomplete_repair_hm(2, 6.0, 2.0)
    assert h2 > 0.1
    assert incomplete_repair_hm(1, 24.0, 2.0) == 0.0


def test_mcnamara_proton_rbe_physics() -> None:
    # Higher LET_d increases RBE; lower alpha/beta increases RBE at fixed LET_d
    rbe_low_let = float(mcnamara_rbe(2.0, alpha_beta_gy=2.0, let_d_kev_um=1.5))
    rbe_high_let = float(mcnamara_rbe(2.0, alpha_beta_gy=2.0, let_d_kev_um=6.0))
    assert rbe_high_let > rbe_low_let > 1.05

    rbe_ab2 = float(mcnamara_rbe(2.0, alpha_beta_gy=2.0, let_d_kev_um=4.0))
    rbe_ab10 = float(mcnamara_rbe(2.0, alpha_beta_gy=10.0, let_d_kev_um=4.0))
    assert rbe_ab2 > rbe_ab10


@pytest.mark.parametrize(
    ("model", "modality", "n_fx"),
    [
        (RadiobiologyModel.LQ, Modality.PHOTON, 25),
        (RadiobiologyModel.LQ, Modality.PROTON_FIXED_RBE, 25),
        (RadiobiologyModel.LQL, Modality.PHOTON, 5),
        (RadiobiologyModel.LEA_CATCHESIDE, Modality.PHOTON, 50),
        (RadiobiologyModel.MCNAMARA_PROTON, Modality.PROTON_VARIABLE_RBE, 25),
    ],
)
def test_exact_forward_inverse_roundtrip_all_models(
    model: RadiobiologyModel, modality: Modality, n_fx: int
) -> None:
    course = CourseSpec(
        course_id="RT",
        label="Roundtrip",
        modality=modality,
        model=model,
        n_fractions=n_fx,
        prescription_gy=40.0,
        d_t_gy=6.0,
        fraction_duration_h=0.5,
        fractions_per_day=2 if model == RadiobiologyModel.LEA_CATCHESIDE else 1,
        interfraction_interval_h=6.0 if model == RadiobiologyModel.LEA_CATCHESIDE else 24.0,
        mean_let_d_kev_um=4.5,
        frame_of_reference_uid=UID,
    )
    target_eqd2 = np.array([0.0, 0.5, 5.0, 15.0, 32.5, 60.0, 110.0], dtype=np.float64)
    let_arr = np.linspace(1.0, 8.0, target_eqd2.size)
    inv_phys = eqd2_to_physical(
        target_eqd2, course, alpha_beta_gy=2.0, repair_half_time_h=2.0, let_d_kev_um=let_arr
    )
    recomputed = physical_to_eqd2(
        inv_phys, course, alpha_beta_gy=2.0, repair_half_time_h=2.0, let_d_kev_um=let_arr
    )
    max_err = float(np.max(np.abs(recomputed - target_eqd2)))
    assert max_err < 1e-10, f"Model {model} roundtrip residual {max_err:.2e} Gy exceeded 1e-10 Gy"


def test_conservative_recovery_guardrails() -> None:
    cord = get_oar_constraint("SpinalCord")
    assert cord is not None
    # Strictly zero recovery before 6 months
    assert conservative_recovery_fraction(cord, months_elapsed=3.0) == 0.0
    assert conservative_recovery_fraction(cord, months_elapsed=5.9) == 0.0
    # Refuses forced non-zero recovery before min_recovery_months
    with pytest.raises(ValueError, match="Cannot apply non-zero recovery"):
        conservative_recovery_fraction(cord, months_elapsed=4.0, override_fraction=0.20)
    # Caps at max_recovery_fraction at >= 24 months
    assert math.isclose(
        conservative_recovery_fraction(cord, months_elapsed=36.0),
        cord.max_recovery_fraction,
        abs_tol=1e-12,
    )
