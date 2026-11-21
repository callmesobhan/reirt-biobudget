"""Clinical DICOM-RT Interoperability: Deformable REG IOD, RTDOSE & Iso-Budget RTSTRUCT Export.

Implements standards-compliant DICOM PS3.3 I/O for radiotherapy TPS interoperability:
1. Deformable Spatial Registration IOD (`1.2.840.10008.5.1.4.1.1.66.3`, Modality='REG'):
   Reads and writes 3D displacement vector fields (DVFs) via `DeformableRegistrationSequence`
   -> `DeformableRegistrationGridSequence` -> `VectorGridData` (OF 32-bit float, mm).
2. RT Dose Storage IOD (`1.2.840.10008.5.1.4.1.1.481.2`, Modality='RTDOSE'):
   Exports 3D voxel-wise remaining physical dose ceiling maps D_rem(x) in Gy for import
   into clinical research TPS planning systems.
3. RT Structure Set Storage IOD (`1.2.840.10008.5.1.4.1.1.481.3`, Modality='RTSTRUCT'):
   Extracts closed axial iso-ceiling contours (e.g., <= 2 Gy, <= 10 Gy, <= 20 Gy remaining
   budget) via marching squares (`contourpy`) and writes DICOM closed-planar ROI contours.

NOT A MEDICAL DEVICE. For academic research and interoperability benchmarking only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import contourpy
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from reirt_biobudget.ir import Volume3D

DEFORMABLE_SPATIAL_REG_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.66.3"


@dataclass
class DICOMDeformableReg:
    """Loaded 3D DICOM Deformable Spatial Registration vector field and frame metadata."""

    dvf_mm: np.ndarray  # canonical (3, Z, Y, X) in mm: (uz, uy, ux)
    spacing_mm: tuple[float, float, float]  # (dz, dy, dx)
    origin_mm: tuple[float, float, float]  # (oz, oy, ox)
    target_frame_of_reference_uid: str
    source_frame_of_reference_uid: str
    registration_description: str


def write_dicom_deformable_reg(
    dvf_mm: np.ndarray,
    spacing_mm: tuple[float, float, float],
    origin_mm: tuple[float, float, float],
    target_frame_of_reference_uid: str,
    source_frame_of_reference_uid: str,
    out_path: str | Path,
    *,
    patient_id: str = "REIRT_RESEARCH",
    description: str = "reirt-biobudget SVF diffeomorphic registration",
) -> Path:
    """Write a canonical (3, Z, Y, X) DVF (uz, uy, ux in mm) as a DICOM Deformable REG file."""
    if dvf_mm.ndim != 4 or dvf_mm.shape[0] != 3:
        raise ValueError(f"Expected dvf_mm shape (3, Z, Y, X), got {dvf_mm.shape}")
    _, nz, ny, nx = dvf_mm.shape
    dz, dy, dx = spacing_mm
    oz, oy, ox = origin_mm

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.UID(DEFORMABLE_SPATIAL_REG_SOP_CLASS)
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(dest), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "REG"
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = target_frame_of_reference_uid
    ds.ContentDate = datetime.now(UTC).strftime("%Y%m%d")
    ds.ContentTime = datetime.now(UTC).strftime("%H%M%S")
    ds.SeriesDescription = description

    # DICOM PS3.3 C.20.3.1.2: VectorGridData is ordered (Z, Y, X) with 3-vector (ux, uy, uz)
    vec_xyz = np.empty((nz, ny, nx, 3), dtype="<f4")
    vec_xyz[..., 0] = dvf_mm[2].astype(np.float32)  # ux
    vec_xyz[..., 1] = dvf_mm[1].astype(np.float32)  # uy
    vec_xyz[..., 2] = dvf_mm[0].astype(np.float32)  # uz

    grid_item = Dataset()
    grid_item.ImagePositionPatient = [float(ox), float(oy), float(oz)]
    grid_item.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    grid_item.GridDimensions = [int(nx), int(ny), int(nz)]
    grid_item.GridResolution = [float(dx), float(dy), float(dz)]
    grid_item.VectorGridData = vec_xyz.tobytes()

    reg_item = Dataset()
    reg_item.SourceFrameOfReferenceUID = source_frame_of_reference_uid
    reg_item.DeformableRegistrationGridSequence = Sequence([grid_item])

    ds.DeformableRegistrationSequence = Sequence([reg_item])
    ds.save_as(str(dest), enforce_file_format=True)
    return dest


def load_dicom_deformable_reg(reg_path: str | Path) -> DICOMDeformableReg:
    """Load a DICOM Deformable Spatial Registration IOD (`REG`) into canonical (3, Z, Y, X) DVF."""
    ds = pydicom.dcmread(str(reg_path))
    if getattr(ds, "Modality", "") != "REG":
        raise ValueError(f"Expected Modality='REG', got '{getattr(ds, 'Modality', '')}'")
    if not hasattr(ds, "DeformableRegistrationSequence") or not ds.DeformableRegistrationSequence:
        raise ValueError("DICOM REG file lacks DeformableRegistrationSequence")

    target_uid = str(getattr(ds, "FrameOfReferenceUID", "")).strip()
    if not target_uid:
        raise ValueError("DICOM REG missing target FrameOfReferenceUID")

    reg_item = ds.DeformableRegistrationSequence[0]
    source_uid = str(getattr(reg_item, "SourceFrameOfReferenceUID", "")).strip()
    if not source_uid:
        raise ValueError("DICOM REG missing SourceFrameOfReferenceUID")

    grid_seq = getattr(reg_item, "DeformableRegistrationGridSequence", None)
    if not grid_seq:
        raise ValueError("DICOM REG missing DeformableRegistrationGridSequence")
    grid = grid_seq[0]

    nx, ny, nz = [int(v) for v in grid.GridDimensions]
    dx, dy, dz = [float(v) for v in grid.GridResolution]
    ox, oy, oz = [float(v) for v in grid.ImagePositionPatient]

    raw_bytes = bytes(grid.VectorGridData)
    vec_xyz = np.frombuffer(raw_bytes, dtype="<f4").reshape((nz, ny, nx, 3))

    dvf_zyx = np.empty((3, nz, ny, nx), dtype=np.float64)
    dvf_zyx[0] = vec_xyz[..., 2]  # uz
    dvf_zyx[1] = vec_xyz[..., 1]  # uy
    dvf_zyx[2] = vec_xyz[..., 0]  # ux

    return DICOMDeformableReg(
        dvf_mm=dvf_zyx,
        spacing_mm=(dz, dy, dx),
        origin_mm=(oz, oy, ox),
        target_frame_of_reference_uid=target_uid,
        source_frame_of_reference_uid=source_uid,
        registration_description=str(getattr(ds, "SeriesDescription", "")),
    )


def export_remaining_budget_dicom_rtdose(
    budget_vol: Volume3D,
    out_path: str | Path,
    *,
    patient_id: str = "REIRT_RESEARCH",
    series_description: str = "reirt-biobudget 3D Remaining Physical Dose Ceiling (p95, Gy)",
) -> Path:
    """Export a 3D remaining physical dose ceiling Volume3D as a standard DICOM RTDOSE file."""
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = budget_vol.shape
    dz, dy, dx = budget_vol.spacing_mm
    oz, oy, ox = budget_vol.origin_mm

    data = np.maximum(np.asarray(budget_vol.data, dtype=np.float64), 0.0)
    max_val = float(np.max(data))
    # Use 32-bit unsigned grid scaling if needed; 16-bit gives < 0.002 Gy resolution up to 130 Gy
    scale = max(max_val / 60000.0, 1e-4)
    arr_u16 = np.clip(np.round(data / scale), 0, 65535).astype("<u2")

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.RTDoseStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(dest), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = budget_vol.frame_of_reference_uid
    ds.SeriesDescription = series_description

    ds.Rows = ny
    ds.Columns = nx
    ds.NumberOfFrames = nz
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [float(dy), float(dx)]
    ds.SliceThickness = float(dz)
    ds.ImagePositionPatient = [float(ox), float(oy), float(oz)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.GridFrameOffsetVector = [float(i * dz) for i in range(nz)]
    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"
    ds.DoseGridScaling = float(scale)
    ds.PixelData = arr_u16.tobytes()

    ds.save_as(str(dest), enforce_file_format=True)
    return dest


def export_budget_iso_rtstruct(
    budget_vol: Volume3D,
    body_mask: np.ndarray,
    out_path: str | Path,
    *,
    iso_thresholds_gy: dict[str, float] | None = None,
    patient_id: str = "REIRT_RESEARCH",
) -> Path:
    """Export closed axial DICOM RTSTRUCT contours marking depleted/tight remaining-budget zones.

    Each ROI marks voxels inside `body_mask` where `remaining_physical_budget <= threshold_gy`,
    providing direct TPS optimization avoidance shells.
    """
    if iso_thresholds_gy is None:
        iso_thresholds_gy = {
            "RemBudget_LE_05Gy": 5.0,
            "RemBudget_LE_15Gy": 15.0,
            "RemBudget_LE_25Gy": 25.0,
        }

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    nz, ny, nx = budget_vol.shape
    dz, dy, dx = budget_vol.spacing_mm
    oz, oy, ox = budget_vol.origin_mm

    x_coords = ox + np.arange(nx, dtype=np.float64) * dx
    y_coords = oy + np.arange(ny, dtype=np.float64) * dy

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(dest), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTSTRUCT"
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StructureSetLabel = "REIRT_BIOBUDGET"
    ds.StructureSetName = "reirt-biobudget Remaining Physical Budget Shells"

    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = budget_vol.frame_of_reference_uid
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])

    ss_rois: list[Dataset] = []
    roi_contours: list[Dataset] = []
    roi_colors = [[213, 94, 0], [230, 159, 0], [0, 114, 178], [0, 158, 115]]

    for roi_idx, (roi_name, thresh_gy) in enumerate(iso_thresholds_gy.items(), start=1):
        roi_item = Dataset()
        roi_item.ROINumber = roi_idx
        roi_item.ReferencedFrameOfReferenceUID = budget_vol.frame_of_reference_uid
        roi_item.ROIName = roi_name
        roi_item.ROIGenerationAlgorithm = "AUTOMATIC"
        ss_rois.append(roi_item)

        mask_3d = body_mask & (budget_vol.data <= float(thresh_gy))
        contours_seq: list[Dataset] = []
        for z_i in range(nz):
            sl = mask_3d[z_i]
            if not np.any(sl):
                continue
            z_mm = oz + z_i * dz
            cg = contourpy.contour_generator(x=x_coords, y=y_coords, z=sl.astype(np.float64))
            lines = cg.lines(0.5)
            for poly in lines:
                if poly.shape[0] < 3:
                    continue
                # Flatten (x, y, z) triplets
                pts_3d = np.column_stack([poly[:, 0], poly[:, 1], np.full(poly.shape[0], z_mm)])
                c_ds = Dataset()
                c_ds.ContourGeometricType = "CLOSED_PLANAR"
                c_ds.NumberOfContourPoints = int(poly.shape[0])
                c_ds.ContourData = [round(float(v), 3) for v in pts_3d.ravel()]
                contours_seq.append(c_ds)

        rc_item = Dataset()
        rc_item.ReferencedROINumber = roi_idx
        rc_item.ROIDisplayColor = roi_colors[(roi_idx - 1) % len(roi_colors)]
        rc_item.ContourSequence = Sequence(contours_seq)
        roi_contours.append(rc_item)

    ds.StructureSetROISequence = Sequence(ss_rois)
    ds.ROIContourSequence = Sequence(roi_contours)
    ds.save_as(str(dest), enforce_file_format=True)
    return dest
