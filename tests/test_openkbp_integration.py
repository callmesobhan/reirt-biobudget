"""Integration test on real open-access OpenKBP clinical patient pt_1 (CC BY 4.0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reirt_biobudget.budget_solver import solve_reirradiation_budget
from reirt_biobudget.io_loaders import (
    build_reirradiation_benchmark_from_openkbp,
    load_openkbp_patient,
)
from reirt_biobudget.ir import ReIrradiationType

PT1_DIR = Path(__file__).resolve().parents[1] / "data" / "openkbp" / "pt_1"


@pytest.mark.skipif(not PT1_DIR.exists(), reason="OpenKBP pt_1 sample not downloaded")
def test_openkbp_pt1_end_to_end_budget_and_invariants() -> None:
    patient = load_openkbp_patient(PT1_DIR)
    assert patient.ct_hu.shape == (128, 128, 128)
    assert patient.dose_gy.shape == (128, 128, 128)
    assert "SpinalCord" in patient.oar_masks
    assert "Brainstem" in patient.oar_masks
    assert "Bone_Mandible" in patient.oar_masks

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
        tre_sigma_mm=2.5,
        n_ensemble=8,
        seed=42,
    )
    res = out.result
    assert res.estro_eortc_classification == ReIrradiationType.TYPE_1_OVERLAP
    assert res.dir_quality.folded_voxel_count == 0
    assert res.dir_quality.tg132_pass is True
    assert res.provenance.all_invariants_passed is True
    assert res.provenance.max_inverse_residual_gy < 1e-9
    assert len(res.oar_summaries) >= 4
