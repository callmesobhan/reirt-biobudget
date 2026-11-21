"""Tests for K-course multi-prior accumulation and clinical DICOM REG/RTDOSE/RTSTRUCT I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest

from reirt_biobudget.budget_solver import solve_multicourse_reirradiation_budget
from reirt_biobudget.dicom_io import (
    DEFORMABLE_SPATIAL_REG_SOP_CLASS,
    export_budget_iso_rtstruct,
    export_remaining_budget_dicom_rtdose,
    load_dicom_deformable_reg,
    write_dicom_deformable_reg,
)
from reirt_biobudget.io_loaders import load_dicom_rt_dose_and_structs
from reirt_biobudget.ir import (
    AbstentionError,
    CourseSpec,
    Modality,
    PriorCourseInput,
    RadiobiologyModel,
    Volume3D,
)


def _make_k2_synthetic() -> tuple[list[PriorCourseInput], CourseSpec, dict[str, np.ndarray], np.ndarray]:
    shape = (18, 18, 18)
    spacing = (2.5, 2.5, 2.5)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]) * spacing[0],
        np.arange(shape[1]) * spacing[1],
        np.arange(shape[2]) * spacing[2],
        indexing="ij",
    )
    r2 = (zz - 22.5) ** 2 + (yy - 22.5) ** 2 + (xx - 22.5) ** 2
    d1 = Volume3D(
        data=40.0 * np.exp(-r2 / (2 * 10.0**2)),
        spacing_mm=spacing,
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.1001",
        name="prior1",
    )
    d2 = Volume3D(
        data=18.0 * np.exp(-r2 / (2 * 7.0**2)),
        spacing_mm=spacing,
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.1002",
        name="prior2",
    )
    c1 = CourseSpec(
        course_id="PRIOR_1",
        label="40/20 IMRT",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQ,
        n_fractions=20,
        prescription_gy=40.0,
        months_before_reirt=30.0,
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.1001",
    )
    c2 = CourseSpec(
        course_id="PRIOR_2",
        label="18/3 SBRT",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQL,
        n_fractions=3,
        prescription_gy=18.0,
        months_before_reirt=12.0,
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.1002",
    )
    planned = CourseSpec(
        course_id="PLANNED_3",
        label="30/5 LQL",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQL,
        n_fractions=5,
        prescription_gy=30.0,
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.1003",
    )
    zero_dvf = np.zeros((3, *shape), dtype=np.float64)
    cord_mask = r2 <= 6.0**2
    body_mask = r2 <= 18.0**2
    priors = [
        PriorCourseInput(course=c1, dose=d1, nominal_dvf_mm=zero_dvf, tre_sigma_mm=2.0),
        PriorCourseInput(course=c2, dose=d2, nominal_dvf_mm=zero_dvf, tre_sigma_mm=1.5),
    ]
    return priors, planned, {"SpinalCord": cord_mask}, body_mask


def test_k2_multicourse_accumulation_and_invariants() -> None:
    priors, planned, oars, body = _make_k2_synthetic()
    out = solve_multicourse_reirradiation_budget(
        case_id="K2_SYNTH",
        prior_inputs=priors,
        planned_course=planned,
        oar_masks_course2=oars,
        body_mask_course2=body,
        n_ensemble=12,
        seed=7,
    )
    res = out.result
    assert len(res.prior_courses) == 2
    assert len(res.per_course_dir_quality) == 2
    assert res.provenance.all_invariants_passed is True
    assert res.provenance.max_inverse_residual_gy < 1e-10
    assert "MULTICOURSE_K2_PAIRED_ENSEMBLE_SUM" in res.provenance.invariants_checked

    # Adding Course 2 must reduce remaining physical ceiling compared to Course 1 alone
    out_k1 = solve_multicourse_reirradiation_budget(
        case_id="K1_ONLY",
        prior_inputs=[priors[0]],
        planned_course=planned,
        oar_masks_course2=oars,
        body_mask_course2=body,
        n_ensemble=12,
        seed=7,
    )
    rem_k2 = out.result.oar_summaries[0].remaining_physical_p95_gy
    rem_k1 = out_k1.result.oar_summaries[0].remaining_physical_p95_gy
    assert rem_k2 < rem_k1


def test_k2_abstains_if_second_prior_course_dvf_folds() -> None:
    priors, planned, oars, body = _make_k2_synthetic()
    folded = np.zeros_like(priors[1].nominal_dvf_mm)
    xx = np.arange(folded.shape[3], dtype=np.float64) * 2.5
    folded[2] = -3.0 * xx[np.newaxis, np.newaxis, :]
    priors[1].nominal_dvf_mm = folded

    with pytest.raises(AbstentionError) as exc:
        solve_multicourse_reirradiation_budget(
            case_id="K2_FOLDED",
            prior_inputs=priors,
            planned_course=planned,
            oar_masks_course2=oars,
            body_mask_course2=body,
            n_ensemble=4,
        )
    assert exc.value.code == "DIR_TOPOLOGY_FOLDING"


def test_dicom_deformable_reg_rtdose_and_rtstruct_roundtrip(tmp_path: Path) -> None:
    priors, planned, oars, body = _make_k2_synthetic()
    out = solve_multicourse_reirradiation_budget(
        case_id="DCM_RT_EXPORT",
        prior_inputs=priors,
        planned_course=planned,
        oar_masks_course2=oars,
        body_mask_course2=body,
        n_ensemble=8,
        seed=42,
    )

    # 1. DICOM Deformable REG write -> read roundtrip
    rng = np.random.default_rng(123)
    dvf = rng.normal(0.0, 2.0, size=(3, 18, 18, 18))
    reg_path = write_dicom_deformable_reg(
        dvf,
        spacing_mm=(2.5, 2.5, 2.5),
        origin_mm=(-20.0, -15.0, -10.0),
        target_frame_of_reference_uid=planned.frame_of_reference_uid,
        source_frame_of_reference_uid=priors[0].course.frame_of_reference_uid,
        out_path=tmp_path / "deformable.dcm",
    )
    ds_raw = pydicom.dcmread(str(reg_path))
    assert str(ds_raw.SOPClassUID) == DEFORMABLE_SPATIAL_REG_SOP_CLASS

    loaded_reg = load_dicom_deformable_reg(reg_path)
    assert loaded_reg.target_frame_of_reference_uid == planned.frame_of_reference_uid
    assert loaded_reg.source_frame_of_reference_uid == priors[0].course.frame_of_reference_uid
    assert loaded_reg.spacing_mm == (2.5, 2.5, 2.5)
    assert loaded_reg.origin_mm == (-20.0, -15.0, -10.0)
    assert np.allclose(loaded_reg.dvf_mm, dvf, atol=1e-5)

    # 2. DICOM RTDOSE remaining budget export -> read roundtrip
    budget_vol = out.remaining_physical_p95_map
    dose_path = export_remaining_budget_dicom_rtdose(budget_vol, tmp_path / "rem_budget_rtdose.dcm")
    # 3. DICOM RTSTRUCT iso-budget shells export
    struct_path = export_budget_iso_rtstruct(
        budget_vol,
        body,
        tmp_path / "rem_budget_rtstruct.dcm",
        iso_thresholds_gy={"RemBudget_LE_25Gy": 25.0, "RemBudget_LE_40Gy": 40.0},
    )

    loaded_dose, loaded_masks = load_dicom_rt_dose_and_structs(dose_path, struct_path)
    assert loaded_dose.shape == budget_vol.shape
    assert np.max(np.abs(loaded_dose.data - budget_vol.data)) < 0.005
    assert loaded_dose.frame_of_reference_uid == budget_vol.frame_of_reference_uid
    assert len(loaded_masks) >= 1
