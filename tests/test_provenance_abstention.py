"""Tests for hard provenance abstention gates and cryptographic SHA-256 ledger integrity."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from reirt_biobudget.budget_solver import solve_reirradiation_budget
from reirt_biobudget.ir import AbstentionError, CourseSpec, Volume3D
from reirt_biobudget.provenance import sha256_volume


def _tiny_problem() -> tuple[CourseSpec, CourseSpec, Volume3D, dict[str, np.ndarray]]:
    shape = (16, 16, 16)
    spacing = (2.5, 2.5, 2.5)
    dose = np.full(shape, 35.0, dtype=np.float64)
    vol = Volume3D(data=dose, spacing_mm=spacing, frame_of_reference_uid="1.2.3.4.C1")
    c1 = CourseSpec(
        course_id="C1",
        label="Prior",
        n_fractions=25,
        prescription_gy=50.0,
        months_before_reirt=12.0,
        frame_of_reference_uid="1.2.3.4.C1",
    )
    c2 = CourseSpec(
        course_id="C2",
        label="Planned",
        n_fractions=5,
        prescription_gy=30.0,
        frame_of_reference_uid="1.2.3.4.C2",
    )
    mask = np.zeros(shape, dtype=bool)
    mask[6:10, 6:10, 6:10] = True
    return c1, c2, vol, {"SpinalCord": mask}


def test_abstains_on_non_diffeomorphic_folded_dvf() -> None:
    c1, c2, vol, oars = _tiny_problem()
    # Construct a deliberately folded DVF (du_x / dx = -3.0 => J = -2.0 < 0!)
    dvf_folded = np.zeros((3, *vol.shape), dtype=np.float64)
    xx = np.arange(vol.shape[2], dtype=np.float64) * vol.spacing_mm[2]
    dvf_folded[2] = -3.0 * xx[np.newaxis, np.newaxis, :]

    with pytest.raises(AbstentionError) as exc_info:
        solve_reirradiation_budget(
            case_id="FOLDED_TEST",
            prior_course=c1,
            prior_dose=vol,
            planned_course=c2,
            nominal_dvf_mm=dvf_folded,
            oar_masks_course2=oars,
            tre_sigma_mm=0.0,
            n_ensemble=1,
        )
    assert exc_info.value.code == "DIR_TOPOLOGY_FOLDING"


def test_abstains_on_excessive_ice() -> None:
    c1, c2, vol, oars = _tiny_problem()
    zero_dvf = np.zeros((3, *vol.shape), dtype=np.float64)
    # Supply deliberately inconsistent inverse DVF (8 mm mismatch!)
    bad_inv = np.full((3, *vol.shape), 6.0, dtype=np.float64)

    with pytest.raises(AbstentionError) as exc_info:
        solve_reirradiation_budget(
            case_id="BAD_ICE_TEST",
            prior_course=c1,
            prior_dose=vol,
            planned_course=c2,
            nominal_dvf_mm=zero_dvf,
            nominal_inv_dvf_mm=bad_inv,
            oar_masks_course2=oars,
            tre_sigma_mm=0.0,
            n_ensemble=1,
            hard_ice_abstention_mm=5.0,
        )
    assert exc_info.value.code == "DIR_EXCESSIVE_ICE"


def test_rejects_unspecified_frame_of_reference_uid() -> None:
    with pytest.raises(ValidationError):
        CourseSpec(
            course_id="BAD",
            label="Missing UID",
            n_fractions=25,
            prescription_gy=50.0,
            frame_of_reference_uid="UNSPECIFIED",
        )


def test_sha256_volume_tamper_detection() -> None:
    _, _, vol, _ = _tiny_problem()
    h1 = sha256_volume(vol)
    vol_mod = Volume3D(
        data=vol.data.copy(),
        spacing_mm=vol.spacing_mm,
        frame_of_reference_uid=vol.frame_of_reference_uid,
    )
    vol_mod.data[8, 8, 8] += 1e-3
    h2 = sha256_volume(vol_mod)
    assert h1 != h2
