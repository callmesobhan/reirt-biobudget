"""Data Loaders: OpenKBP Public Clinical Dataset (CC BY 4.0) & DICOM-RT Ingestion.

Supports:
1. OpenKBP sparse CSV loader for real de-identified head-and-neck IMRT CT, 3D clinical
   dose distributions, and contoured OAR/PTV masks (Babier et al., Med Phys 2021;48:5549,
   DOI: 10.1002/mp.14845, License: CC BY 4.0 / MIT).
2. Reproducible longitudinal re-irradiation benchmark generator built on top of real
   OpenKBP Course-1 anatomy & dose with explicitly documented, diffeomorphic SVF
   follow-up anatomy shifts and candidate salvage Course-2 dose distributions.
3. Standard DICOM-RT loader (CT + RTDOSE + RTSTRUCT via pydicom) with FrameOfReferenceUID
   validation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt, gaussian_filter

from reirt_biobudget.dir_envelope import integrate_svf_to_dvf, warp_array
from reirt_biobudget.ir import CourseSpec, Modality, RadiobiologyModel, Volume3D
from reirt_biobudget.ontology import normalize_tg263

OPENKBP_SHAPE = (128, 128, 128)
OPENKBP_OAR_NAMES = [
    "Brainstem",
    "SpinalCord",
    "RightParotid",
    "LeftParotid",
    "Esophagus",
    "Larynx",
    "Mandible",
]
OPENKBP_PTV_NAMES = ["PTV70", "PTV63", "PTV56"]


@dataclass
class OpenKBPPatient:
    """Real de-identified OpenKBP patient volumes in canonical (z=SI, y=AP, x=RL) orientation."""

    patient_id: str
    ct_hu: Volume3D
    dose_gy: Volume3D
    body_mask: np.ndarray
    oar_masks: dict[str, np.ndarray]  # Keys normalized to AAPM TG-263
    ptv_masks: dict[str, np.ndarray]
    manifest: dict[str, Any]


@dataclass
class ReIrradiationBenchmarkCase:
    """Complete two-course re-irradiation benchmark derived from a real OpenKBP clinical plan."""

    case_id: str
    patient_id: str
    prior_course: CourseSpec
    planned_course: CourseSpec
    ct_course1: Volume3D
    ct_course2: Volume3D
    prior_dose_course1: Volume3D
    candidate_dose_course2: Volume3D
    let_d_course2: Volume3D
    nominal_dvf_mm: np.ndarray  # Maps Course-2 grid -> Course-1 grid
    nominal_inv_dvf_mm: np.ndarray  # Exact analytical SVF inverse: Maps Course-1 -> Course-2
    body_mask_course2: np.ndarray
    oar_masks_course1: dict[str, np.ndarray]
    oar_masks_course2: dict[str, np.ndarray]
    recurrent_ptv_course2: np.ndarray
    dataset_source: str
    dataset_citation: str
    dataset_license: str


def _load_sparse_csv(path: Path, is_binary: bool = False) -> np.ndarray:
    """Load an OpenKBP flattened sparse CSV into (128, 128, 128) raw order (AP, RL, SI)."""
    flat = np.zeros(int(np.prod(OPENKBP_SHAPE)), dtype=np.float64)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header: ,data
        if is_binary:
            indices = [int(row[0]) for row in reader if row and row[0]]
            if indices:
                flat[np.asarray(indices, dtype=np.int64)] = 1.0
        else:
            indices = []
            values = []
            for row in reader:
                if row and row[0] and len(row) > 1 and row[1]:
                    indices.append(int(row[0]))
                    values.append(float(row[1]))
            if indices:
                flat[np.asarray(indices, dtype=np.int64)] = np.asarray(values, dtype=np.float64)
    return flat.reshape(OPENKBP_SHAPE)


def _to_canonical_zyx(arr_ap_rl_si: np.ndarray) -> np.ndarray:
    """Transpose raw OpenKBP (AP, RL, SI) -> canonical radiological (z=SI, y=AP, x=RL)."""
    return np.ascontiguousarray(np.transpose(arr_ap_rl_si, (2, 0, 1)))


def load_openkbp_patient(patient_dir: str | Path) -> OpenKBPPatient:
    """Load a real OpenKBP patient directory into canonical Volume3D containers."""
    pdir = Path(patient_dir)
    if not (pdir / "ct.csv").exists() or not (pdir / "dose.csv").exists():
        raise FileNotFoundError(
            f"OpenKBP patient files not found in {pdir}. "
            "Run `python scripts/fetch_openkbp_sample.py --patient pt_1` first."
        )

    vox_raw = np.loadtxt(pdir / "voxel_dimensions.csv", dtype=np.float64)
    # Raw OpenKBP voxel_dimensions.csv order is [AP_mm, RL_mm, SI_mm]
    spacing_zyx = (float(vox_raw[2]), float(vox_raw[0]), float(vox_raw[1]))

    ct_raw = _to_canonical_zyx(_load_sparse_csv(pdir / "ct.csv", is_binary=False))
    # OpenKBP shifts CT by +1000 HU (air=0, water~1000)
    ct_hu_arr = ct_raw - 1000.0
    dose_arr = _to_canonical_zyx(_load_sparse_csv(pdir / "dose.csv", is_binary=False))

    if (pdir / "possible_dose_mask.csv").exists():
        body_mask = _to_canonical_zyx(
            _load_sparse_csv(pdir / "possible_dose_mask.csv", is_binary=True)
        ) > 0.5
    else:
        body_mask = ct_hu_arr > -500.0

    for_uid = f"1.2.826.0.1.3680043.8.498.OPENKBP.{pdir.name}.COURSE1"

    ct_vol = Volume3D(
        data=ct_hu_arr,
        spacing_mm=spacing_zyx,
        origin_mm=(0.0, 0.0, 0.0),
        frame_of_reference_uid=for_uid,
        units="HU",
        name=f"{pdir.name}_ct",
    )
    dose_vol = Volume3D(
        data=dose_arr,
        spacing_mm=spacing_zyx,
        origin_mm=(0.0, 0.0, 0.0),
        frame_of_reference_uid=for_uid,
        units="Gy",
        name=f"{pdir.name}_dose",
    )

    oar_masks: dict[str, np.ndarray] = {}
    for raw_oar in OPENKBP_OAR_NAMES:
        fpath = pdir / f"{raw_oar}.csv"
        if fpath.exists():
            mask = _to_canonical_zyx(_load_sparse_csv(fpath, is_binary=True)) > 0.5
            if np.any(mask):
                oar_masks[normalize_tg263(raw_oar)] = mask

    ptv_masks: dict[str, np.ndarray] = {}
    for ptv_name in OPENKBP_PTV_NAMES:
        fpath = pdir / f"{ptv_name}.csv"
        if fpath.exists():
            mask = _to_canonical_zyx(_load_sparse_csv(fpath, is_binary=True)) > 0.5
            if np.any(mask):
                ptv_masks[ptv_name] = mask

    manifest: dict[str, Any] = {}
    manifest_path = pdir / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return OpenKBPPatient(
        patient_id=pdir.name,
        ct_hu=ct_vol,
        dose_gy=dose_vol,
        body_mask=body_mask,
        oar_masks=oar_masks,
        ptv_masks=ptv_masks,
        manifest=manifest,
    )


def build_reirradiation_benchmark_from_openkbp(
    patient: OpenKBPPatient,
    *,
    months_elapsed: float = 18.0,
    planned_n_fractions: int = 5,
    planned_rx_gy: float = 35.0,
    planned_modality: Modality = Modality.PHOTON,
    planned_model: RadiobiologyModel = RadiobiologyModel.LQL,
) -> ReIrradiationBenchmarkCase:
    """Build a realistic longitudinal H&N re-irradiation case from real OpenKBP Course-1 data.

    Course 1 uses verbatim OpenKBP clinical CT, 3D IMRT dose (70 Gy / 35 fx), and contoured
    OAR masks. Course 2 simulates an 18-month post-treatment follow-up anatomy via a smooth,
    invertible Stationary Velocity Field (SVF) modeling mild post-RT neck tissue contraction
    and cervical posture change, plus a realistic salvage recurrent PTV and candidate plan.
    """
    shape = patient.ct_hu.shape
    spacing = patient.ct_hu.spacing_mm
    dz, dy, dx = spacing

    # Construct smooth, physically plausible follow-up SVF (Course 2 -> Course 1 pull-back)
    nz, ny, nx = shape
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64) * dz,
        np.arange(ny, dtype=np.float64) * dy,
        np.arange(nx, dtype=np.float64) * dx,
        indexing="ij",
    )

    # Center coordinates around the neck/PTV70 centroid
    anchor_mask = patient.ptv_masks.get("PTV70", patient.body_mask)
    cz = float(np.mean(zz[anchor_mask]))
    cy = float(np.mean(yy[anchor_mask]))
    cx = float(np.mean(xx[anchor_mask]))

    r2 = ((zz - cz) / 55.0) ** 2 + ((yy - cy) / 48.0) ** 2 + ((xx - cx) / 48.0) ** 2
    gauss_env = np.exp(-0.5 * r2)

    # Smooth post-chemoradiation medial soft-tissue contraction + mild AP cervical flexion (~3-4 mm)
    vel = np.zeros((3, *shape), dtype=np.float64)
    vel[0] = 2.2 * np.sin((zz - cz) / 60.0) * gauss_env  # SI stretch/shift (mm)
    vel[1] = (2.6 + 1.8 * ((yy - cy) / 55.0)) * gauss_env  # AP flexion + anterior shrinkage (mm)
    vel[2] = (3.4 * ((xx - cx) / 45.0)) * gauss_env  # Lateral weight-loss contraction (mm)

    # Integrate SVF via scaling-and-squaring -> exact forward & inverse diffeomorphisms!
    dvf_fwd = integrate_svf_to_dvf(vel, spacing, steps=4)
    dvf_inv = integrate_svf_to_dvf(-vel, spacing, steps=4)

    # Warp Course-1 CT & masks into Course-2 geometry
    ct2_arr = warp_array(patient.ct_hu.data, dvf_fwd, spacing, order=1, cval=-1000.0)
    body2_mask = warp_array(
        patient.body_mask.astype(np.float64), dvf_fwd, spacing, order=1, cval=0.0
    ) > 0.5
    oar_masks_c2: dict[str, np.ndarray] = {}
    for name, mask1 in patient.oar_masks.items():
        warped_m = warp_array(mask1.astype(np.float64), dvf_fwd, spacing, order=1, cval=0.0) > 0.5
        if np.any(warped_m):
            oar_masks_c2[name] = warped_m

    # Construct a realistic recurrent salvage GTV/PTV overlapping prior PTV70 near neck/cord gradient
    ptv70_c2 = (
        warp_array(
            patient.ptv_masks["PTV70"].astype(np.float64), dvf_fwd, spacing, order=1, cval=0.0
        )
        > 0.5
        if "PTV70" in patient.ptv_masks
        else body2_mask
    )
    # Place recurrent salvage PTV (radius ~18 mm) in the posterior-medial quadrant of prior PTV70
    # so it presents a realistic clinical tradeoff with SpinalCord, Brainstem, and Mandible
    if "SpinalCord" in oar_masks_c2:
        cord_z = float(np.mean(zz[oar_masks_c2["SpinalCord"]]))
        cord_y = float(np.mean(yy[oar_masks_c2["SpinalCord"]]))
        cord_x = float(np.mean(xx[oar_masks_c2["SpinalCord"]]))
        # 29 mm anterior & 19 mm left-lateral of upper cervical cord (clinically realistic
        # salvage oropharyngeal/nodal recurrence near cord/mandible dose-sparing frontier)
        rz = 0.58 * cz + 0.42 * cord_z
        ry = cord_y - 29.0
        rx = cord_x + 19.0
    else:
        rz, ry, rx = cz, cy, cx

    dist_rec = np.sqrt(((zz - rz) / 16.0) ** 2 + ((yy - ry) / 16.0) ** 2 + ((xx - rx) / 16.0) ** 2)
    recurrent_ptv = (dist_rec <= 1.0) & body2_mask
    if not np.any(recurrent_ptv):
        recurrent_ptv = ptv70_c2 & body2_mask

    # Generate realistic candidate Course-2 conformal dose distribution (steep SBRT/IMRT gradient)
    # Distance in mm outside recurrent PTV
    dist_outside_mm = distance_transform_edt(~recurrent_ptv, sampling=spacing)
    penumbra_mm = 7.8 if planned_n_fractions <= 10 else 10.8
    raw_c2 = planned_rx_gy * np.exp(-((dist_outside_mm / penumbra_mm) ** 1.45))
    raw_c2 = np.where(recurrent_ptv, planned_rx_gy * 1.05, raw_c2)
    # Slight physical smoothing + zero outside body
    raw_c2 = gaussian_filter(raw_c2, sigma=(0.7, 0.7, 0.7)) * body2_mask.astype(np.float64)

    # Realistic distal-edge proton LET_d map (elevated ~4.5-6.5 keV/um at target distal edge near cord)
    distal_band = np.exp(-0.5 * ((dist_outside_mm - 6.0) / 7.0) ** 2)
    let_d_arr = (2.2 + 3.8 * distal_band) * body2_mask.astype(np.float64)

    for_uid_c1 = patient.ct_hu.frame_of_reference_uid
    for_uid_c2 = f"1.2.826.0.1.3680043.8.498.OPENKBP.{patient.patient_id}.COURSE2"

    prior_course = CourseSpec(
        course_id="COURSE_1_PRIMARY",
        label="Primary H&N IMRT (70 Gy / 35 fx)",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQ,
        n_fractions=35,
        prescription_gy=70.0,
        months_before_reirt=months_elapsed,
        frame_of_reference_uid=for_uid_c1,
        source_provenance=f"OpenKBP ({patient.patient_id}, Babier et al. Med Phys 2021, CC BY 4.0)",
    )

    planned_course = CourseSpec(
        course_id="COURSE_2_SALVAGE",
        label=f"Salvage Re-Irradiation ({planned_rx_gy:.0f} Gy / {planned_n_fractions} fx, {planned_model.value.upper()})",
        modality=planned_modality,
        model=planned_model,
        n_fractions=planned_n_fractions,
        prescription_gy=planned_rx_gy,
        months_before_reirt=0.0,
        frame_of_reference_uid=for_uid_c2,
        source_provenance="Derived longitudinal benchmark (analytical SVF + conformal salvage dose)",
    )

    ct_c2 = Volume3D(
        data=ct2_arr,
        spacing_mm=spacing,
        origin_mm=patient.ct_hu.origin_mm,
        frame_of_reference_uid=for_uid_c2,
        units="HU",
        name=f"{patient.patient_id}_ct_course2",
    )
    cand_c2 = Volume3D(
        data=raw_c2,
        spacing_mm=spacing,
        origin_mm=patient.ct_hu.origin_mm,
        frame_of_reference_uid=for_uid_c2,
        units="Gy",
        name=f"{patient.patient_id}_candidate_dose_course2",
    )
    let_c2 = Volume3D(
        data=let_d_arr,
        spacing_mm=spacing,
        origin_mm=patient.ct_hu.origin_mm,
        frame_of_reference_uid=for_uid_c2,
        units="keV/um",
        name=f"{patient.patient_id}_let_d_course2",
    )

    return ReIrradiationBenchmarkCase(
        case_id=f"OPENKBP-{patient.patient_id.upper()}-REIRT",
        patient_id=patient.patient_id,
        prior_course=prior_course,
        planned_course=planned_course,
        ct_course1=patient.ct_hu,
        ct_course2=ct_c2,
        prior_dose_course1=patient.dose_gy,
        candidate_dose_course2=cand_c2,
        let_d_course2=let_c2,
        nominal_dvf_mm=dvf_fwd,
        nominal_inv_dvf_mm=dvf_inv,
        body_mask_course2=body2_mask,
        oar_masks_course1=patient.oar_masks,
        oar_masks_course2=oar_masks_c2,
        recurrent_ptv_course2=recurrent_ptv,
        dataset_source=f"OpenKBP public clinical dataset ({patient.patient_id})",
        dataset_citation=(
            "Babier A, Zhang B, Mahmood R, Moore KL, Purdie TG, McNiven AL, Chan TCY. "
            "OpenKBP: The open-access knowledge-based planning grand challenge and dataset. "
            "Medical Physics. 2021;48(9):5549-5561. DOI: 10.1002/mp.14845"
        ),
        dataset_license="CC BY 4.0 (dataset) / MIT (repository code)",
    )


def load_dicom_rt_dose_and_structs(
    rtdose_path: str | Path,
    rtstruct_path: str | Path | None = None,
) -> tuple[Volume3D, dict[str, np.ndarray]]:
    """Load standard DICOM-RT RTDOSE (+ optional RTSTRUCT) via pydicom with UID provenance.

    Extracts physical dose in Gy via `pixel_array * DoseGridScaling`, voxel spacing from
    `PixelSpacing` and `GridFrameOffsetVector`, and rasterizes RTSTRUCT contours onto the
    RTDOSE grid.
    """
    import pydicom

    ds_dose = pydicom.dcmread(str(rtdose_path))
    if getattr(ds_dose, "Modality", "") != "RTDOSE":
        raise ValueError(f"Expected Modality='RTDOSE', got '{getattr(ds_dose, 'Modality', '')}'")

    for_uid = str(getattr(ds_dose, "FrameOfReferenceUID", "")).strip()
    if not for_uid:
        raise ValueError("DICOM RTDOSE is missing mandatory FrameOfReferenceUID")

    scale = float(getattr(ds_dose, "DoseGridScaling", 1.0))
    arr = ds_dose.pixel_array.astype(np.float64) * scale
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    py_mm, px_mm = [float(v) for v in ds_dose.PixelSpacing]
    offsets = getattr(ds_dose, "GridFrameOffsetVector", None)
    if offsets is not None and len(offsets) > 1:
        dz_mm = float(abs(float(offsets[1]) - float(offsets[0])))
    else:
        dz_mm = float(getattr(ds_dose, "SliceThickness", 2.5))

    ipp = [float(v) for v in getattr(ds_dose, "ImagePositionPatient", [0.0, 0.0, 0.0])]
    # Origin in (z, y, x) order
    origin_zyx = (ipp[2], ipp[1], ipp[0])
    spacing_zyx = (dz_mm, py_mm, px_mm)

    dose_vol = Volume3D(
        data=arr,
        spacing_mm=spacing_zyx,
        origin_mm=origin_zyx,
        frame_of_reference_uid=for_uid,
        units="Gy",
        name=Path(rtdose_path).stem,
    )

    masks: dict[str, np.ndarray] = {}
    if rtstruct_path is not None:
        ds_struct = pydicom.dcmread(str(rtstruct_path))
        struct_for_uid = ""
        if hasattr(ds_struct, "ReferencedFrameOfReferenceSequence"):
            struct_for_uid = str(
                getattr(ds_struct.ReferencedFrameOfReferenceSequence[0], "FrameOfReferenceUID", "")
            ).strip()
        if struct_for_uid and struct_for_uid != for_uid:
            raise ValueError(
                f"DICOM FrameOfReferenceUID mismatch: RTSTRUCT ({struct_for_uid}) != RTDOSE ({for_uid})"
            )

        roi_names = {
            int(roi.ROINumber): normalize_tg263(str(roi.ROIName))
            for roi in getattr(ds_struct, "StructureSetROISequence", [])
        }
        nz, ny, nx = arr.shape
        yy_grid, xx_grid = np.meshgrid(
            origin_zyx[1] + np.arange(ny) * py_mm,
            origin_zyx[2] + np.arange(nx) * px_mm,
            indexing="ij",
        )
        pts = np.column_stack([xx_grid.ravel(), yy_grid.ravel()])
        z_coords = (
            np.asarray([origin_zyx[0] + float(o) for o in offsets], dtype=np.float64)
            if offsets is not None and len(offsets) == nz
            else origin_zyx[0] + np.arange(nz) * dz_mm
        )

        for rc in getattr(ds_struct, "ROIContourSequence", []):
            name = roi_names.get(int(rc.ReferencedROINumber))
            if not name or not hasattr(rc, "ContourSequence"):
                continue
            mask = np.zeros((nz, ny, nx), dtype=bool)
            for contour in rc.ContourSequence:
                data = np.asarray(contour.ContourData, dtype=np.float64).reshape(-1, 3)
                if data.shape[0] < 3:
                    continue
                z_val = float(data[0, 2])
                z_idx = int(np.argmin(np.abs(z_coords - z_val)))
                if abs(z_coords[z_idx] - z_val) > dz_mm * 0.75:
                    continue
                poly = MplPath(data[:, :2])
                inside = poly.contains_points(pts).reshape(ny, nx)
                mask[z_idx] |= inside
            if np.any(mask):
                masks[name] = mask

    return dose_vol, masks
