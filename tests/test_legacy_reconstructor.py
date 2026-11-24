"""Tests for Missing-DICOM / Legacy-PDF 3D Prior-Dose Envelope Reconstruction & MCP tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reirt_biobudget.dir_envelope import warp_array
from reirt_biobudget.harness.mcp_server import handle_jsonrpc_message
from reirt_biobudget.io_loaders import (
    build_reirradiation_benchmark_from_openkbp,
    load_openkbp_patient,
)
from reirt_biobudget.ir import AbstentionError, CourseSpec, Modality, RadiobiologyModel, Volume3D
from reirt_biobudget.legacy_reconstructor import (
    generate_legacy_summary_text_from_ground_truth,
    parse_legacy_treatment_summary,
    solve_legacy_missing_dicom_budget,
)
from reirt_biobudget.visualization import plot_fig8_missing_dicom_reconstruction

OPENKBP_ROOT = Path(__file__).resolve().parents[1] / "data" / "openkbp"
PT1_DIR = OPENKBP_ROOT / "pt_1"


def test_parse_legacy_summary_source_spans_and_tg263() -> None:
    sample = (
        "Outside Hospital Oncology Note. 6 MV VMAT head-and-neck plan completed 24 months ago.\n"
        "Prescription: 70.0 Gy in 35 fractions. Global Dmax: 74.1 Gy.\n"
        "Spinal Cord Dmax = 32.5 Gy\n"
        "Brainstem D0.03cc: 38.0 Gy\n"
        "Left Parotid Dmean = 42.0 Gy\n"
    )
    rec = parse_legacy_treatment_summary(sample, record_id="OUTSIDE_01")
    assert rec.prescription_gy == 70.0
    assert rec.n_fractions == 35
    assert rec.months_before_reirt == 24.0
    assert rec.global_dmax_gy == 74.1
    assert len(rec.oar_constraints) == 3

    # Verify every source span matches the exact verbatim substring in input text
    for span in rec.evidence_spans:
        assert sample[span.start_char : span.end_char] == span.quote

    oars = {c.tg263_name: c for c in rec.oar_constraints}
    assert oars["SpinalCord"].value == 32.5
    assert oars["Brainstem"].value == 38.0
    assert oars["Parotid_L"].value == 42.0


def test_legacy_parser_hard_abstention_gates() -> None:
    # 1. Missing fractionation -> LEGACY_MISSING_FRACTIONATION
    with pytest.raises(AbstentionError) as exc1:
        parse_legacy_treatment_summary("Patient received IMRT to neck. SpinalCord Dmax = 30 Gy.")
    assert exc1.value.code == "LEGACY_MISSING_FRACTIONATION"

    # 2. Conflicting values for same OAR/metric -> LEGACY_CONFLICTING_FACTS
    with pytest.raises(AbstentionError) as exc2:
        parse_legacy_treatment_summary(
            "70 Gy in 35 fractions. SpinalCord Dmax = 28 Gy. Later note: Spinal Cord Dmax = 44 Gy."
        )
    assert exc2.value.code == "LEGACY_CONFLICTING_FACTS"

    # 3. Unphysical Dmean > Dmax -> LEGACY_UNPHYSICAL_DVH_CONSTRAINT
    with pytest.raises(AbstentionError) as exc3:
        parse_legacy_treatment_summary(
            "60 Gy in 30 fractions. Parotid_L Dmax = 25 Gy. Left Parotid Dmean = 40 Gy."
        )
    assert exc3.value.code == "LEGACY_UNPHYSICAL_DVH_CONSTRAINT"

    # 4. Unphysical OAR dose > 140% Rx -> LEGACY_UNPHYSICAL_DVH_CONSTRAINT
    with pytest.raises(AbstentionError) as exc4:
        parse_legacy_treatment_summary("50 Gy in 25 fractions. SpinalCord Dmax = 95 Gy.")
    assert exc4.value.code == "LEGACY_UNPHYSICAL_DVH_CONSTRAINT"


def test_legacy_reconstructor_abstains_without_anchors() -> None:
    rec = parse_legacy_treatment_summary("60 Gy in 30 fractions. No OAR doses available.")
    body = np.ones((12, 12, 12), dtype=bool)
    planned = CourseSpec(
        course_id="C2",
        label="35/5 LQL",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQL,
        n_fractions=5,
        prescription_gy=35.0,
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.9999.2",
    )
    with pytest.raises(AbstentionError) as exc:
        solve_legacy_missing_dicom_budget(
            case_id="NO_ANCHOR",
            legacy_record=rec,
            planned_course=planned,
            body_mask_current=body,
            oar_masks_current={},
            spacing_mm=(2.5, 2.5, 2.5),
        )
    assert exc.value.code == "LEGACY_NO_ANCHOR_CONSTRAINTS"


@pytest.mark.skipif(not PT1_DIR.exists(), reason="OpenKBP pt_1 sample not downloaded")
def test_blinded_openkbp_missing_dicom_reconstruction_and_fig8(tmp_path: Path) -> None:
    patient = load_openkbp_patient(PT1_DIR)
    case = build_reirradiation_benchmark_from_openkbp(
        patient, months_elapsed=18.0, planned_n_fractions=5, planned_rx_gy=35.0
    )
    warped_truth = warp_array(
        case.prior_dose_course1.data,
        case.nominal_dvf_mm,
        case.ct_course2.spacing_mm,
        order=1,
        cval=0.0,
        clamp_nonnegative=True,
    )
    truth_vol = Volume3D(
        data=warped_truth,
        spacing_mm=case.ct_course2.spacing_mm,
        origin_mm=case.ct_course2.origin_mm,
        frame_of_reference_uid=case.planned_course.frame_of_reference_uid,
        units="Gy",
        name="hidden_truth",
    )
    summary_text = generate_legacy_summary_text_from_ground_truth(
        patient_id=patient.patient_id,
        prior_dose=case.prior_dose_course1,
        oar_masks=case.oar_masks_course1,
    )
    rec = parse_legacy_treatment_summary(summary_text, record_id="LEGACY_PT1")
    approx_ptv = (
        warp_array(
            patient.ptv_masks["PTV70"].astype(float),
            case.nominal_dvf_mm,
            case.ct_course2.spacing_mm,
            order=1,
            cval=0.0,
        )
        > 0.5
    )

    out = solve_legacy_missing_dicom_budget(
        case_id="PT1_NO_DICOM",
        legacy_record=rec,
        planned_course=case.planned_course,
        body_mask_current=case.body_mask_course2,
        oar_masks_current=case.oar_masks_course2,
        spacing_mm=case.ct_course2.spacing_mm,
        origin_mm=case.ct_course2.origin_mm,
        approx_prior_ptv_current=approx_ptv,
        hidden_ground_truth_dose=truth_vol,
    )

    # Strict 3D bound ordering invariant: 0 <= lower <= nominal <= upper everywhere
    assert np.all(out.dose_lower_3d.data >= -1e-9)
    assert np.all(out.dose_lower_3d.data <= out.dose_nominal_3d.data + 1e-6)
    assert np.all(out.dose_nominal_3d.data <= out.dose_upper_3d.data + 1e-6)

    aud = out.audit
    assert aud.max_constraint_fit_error_gy < 0.2
    assert aud.max_inverse_residual_gy < 1e-10
    assert aud.provenance.all_invariants_passed is True
    assert aud.truth_voxel_upper_safety_rate_pct is not None
    assert aud.truth_voxel_upper_safety_rate_pct >= 90.0

    # Every serial OAR conservative upper bound must safely bound hidden OpenKBP truth
    for s in aud.oar_summaries:
        if s.tg263_name in ("SpinalCord", "Brainstem", "Bone_Mandible"):
            assert s.envelope_safely_bounds_truth is True

    # Verify Figure 8 renders cleanly
    f8 = plot_fig8_missing_dicom_reconstruction(
        case, out, truth_vol, [(patient.patient_id, aud)], tmp_path
    )
    assert f8.exists() and f8.stat().st_size > 15_000


def test_mcp_legacy_tools_and_abstention() -> None:
    # 1. Valid parse via MCP
    resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "biobudget_parse_legacy_summary",
                "arguments": {
                    "summary_text": "60 Gy in 30 fractions 24 months ago. SpinalCord Dmax = 31.0 Gy."
                },
            },
        }
    )
    assert resp is not None and resp["result"]["isError"] is False
    parsed = json.loads(resp["result"]["content"][0]["text"])
    assert parsed["prescription_gy"] == 60.0
    assert parsed["n_fractions"] == 30

    # 2. Missing fractionation triggers structured MCP ABSTAIN response
    bad = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "biobudget_parse_legacy_summary",
                "arguments": {"summary_text": "Prior RT to neck. SpinalCord Dmax = 31.0 Gy."},
            },
        }
    )
    assert bad is not None and bad["result"]["isError"] is True
    bad_payload = json.loads(bad["result"]["content"][0]["text"])
    assert bad_payload["status"] == "ABSTAIN"
    assert bad_payload["code"] == "LEGACY_MISSING_FRACTIONATION"

    # 3. End-to-end blinded reconstruction via MCP
    if PT1_DIR.exists():
        rec_resp = handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 23,
                "method": "tools/call",
                "params": {
                    "name": "biobudget_reconstruct_legacy_budget",
                    "arguments": {"patient_dir": str(PT1_DIR)},
                },
            }
        )
        assert rec_resp is not None and rec_resp["result"]["isError"] is False
        rec_data = json.loads(rec_resp["result"]["content"][0]["text"])
        assert rec_data["max_inverse_residual_gy"] < 1e-10
        assert rec_data["truth_voxel_upper_safety_rate_pct"] >= 90.0
