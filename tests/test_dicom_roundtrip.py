"""End-to-end DICOM-RT RTDOSE + RTSTRUCT creation, ingestion, and UID-mismatch verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from reirt_biobudget.io_loaders import load_dicom_rt_dose_and_structs


def _write_synthetic_rtdose_and_rtstruct(
    tmp_path: Path,
    for_uid_dose: str,
    for_uid_struct: str,
) -> tuple[Path, Path]:
    dose_path = tmp_path / "rtdose.dcm"
    struct_path = tmp_path / "rtstruct.dcm"

    # Build valid DICOM RTDOSE (10x16x16, 2 mm isotropic, max 50 Gy)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.RTDoseStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(dose_path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"
    ds.FrameOfReferenceUID = for_uid_dose
    ds.Rows = 16
    ds.Columns = 16
    ds.NumberOfFrames = 10
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [2.0, 2.0]
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.GridFrameOffsetVector = [float(i * 2.0) for i in range(10)]
    ds.DoseUnits = "GY"
    ds.DoseGridScaling = 0.01

    arr = np.zeros((10, 16, 16), dtype=np.uint16)
    arr[3:7, 5:11, 5:11] = 4500  # 45.00 Gy
    ds.PixelData = arr.tobytes()
    ds.save_as(str(dose_path))

    # Build valid DICOM RTSTRUCT with a square SpinalCord contour on z = 8.0 mm (slice 4)
    meta_s = FileMetaDataset()
    meta_s.MediaStorageSOPClassUID = pydicom.uid.RTStructureSetStorage
    meta_s.MediaStorageSOPInstanceUID = generate_uid()
    meta_s.TransferSyntaxUID = ExplicitVRLittleEndian

    ss = FileDataset(str(struct_path), {}, file_meta=meta_s, preamble=b"\0" * 128)
    ss.SOPClassUID = meta_s.MediaStorageSOPClassUID
    ss.SOPInstanceUID = meta_s.MediaStorageSOPInstanceUID
    ss.Modality = "RTSTRUCT"

    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = for_uid_struct
    ss.ReferencedFrameOfReferenceSequence = Sequence([ref_for])

    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "spinal_cord"  # non-standard alias -> should normalize to SpinalCord
    ss.StructureSetROISequence = Sequence([roi])

    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = 4
    # Square in (x, y) from [11, 19] x [11, 19] mm at z = 8.0 mm
    contour.ContourData = [11.0, 11.0, 8.0, 19.0, 11.0, 8.0, 19.0, 19.0, 8.0, 11.0, 19.0, 8.0]

    rc = Dataset()
    rc.ReferencedROINumber = 1
    rc.ContourSequence = Sequence([contour])
    ss.ROIContourSequence = Sequence([rc])
    ss.save_as(str(struct_path))

    return dose_path, struct_path


def test_dicom_rt_roundtrip_and_tg263_normalization(tmp_path: Path) -> None:
    uid = generate_uid()
    dose_p, struct_p = _write_synthetic_rtdose_and_rtstruct(tmp_path, uid, uid)
    vol, masks = load_dicom_rt_dose_and_structs(dose_p, struct_p)

    assert vol.shape == (10, 16, 16)
    assert vol.spacing_mm == (2.0, 2.0, 2.0)
    assert np.isclose(float(np.max(vol.data)), 45.0, atol=1e-6)
    assert "SpinalCord" in masks
    assert int(np.sum(masks["SpinalCord"])) > 0
    # Inside the contour on z=4, dose must equal 45.0 Gy
    assert np.allclose(vol.data[masks["SpinalCord"]], 45.0, atol=1e-6)


def test_dicom_rt_rejects_mismatched_for_uid(tmp_path: Path) -> None:
    dose_p, struct_p = _write_synthetic_rtdose_and_rtstruct(
        tmp_path, generate_uid(), generate_uid()
    )
    with pytest.raises(ValueError, match="FrameOfReferenceUID mismatch"):
        load_dicom_rt_dose_and_structs(dose_p, struct_p)
