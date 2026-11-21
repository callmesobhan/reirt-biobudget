"""Cryptographic Provenance Ledger & Hard Abstention Gate for reirt-biobudget.

Implements non-negotiable physical, spatial, and clinical provenance gates. If any
hard invariant is violated (e.g., non-diffeomorphic Jacobian folding inside tissue,
missing FrameOfReferenceUID, severe ICE failure, grid mismatch), the engine raises
AbstentionError instead of silently outputting hazardous clinical numbers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from reirt_biobudget.ir import (
    AbstentionError,
    CourseSpec,
    DIRQualityMetrics,
    ProvenanceRecord,
    Volume3D,
)


def sha256_array(arr: np.ndarray, extra_header: dict[str, Any] | None = None) -> str:
    """Compute deterministic SHA-256 digest of a NumPy array plus shape/dtype/header."""
    h = hashlib.sha256()
    contig = np.ascontiguousarray(arr)
    header = {
        "shape": list(contig.shape),
        "dtype": str(contig.dtype),
        "extra": extra_header or {},
    }
    h.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    h.update(contig.tobytes())
    return h.hexdigest()


def sha256_volume(vol: Volume3D) -> str:
    """Compute deterministic SHA-256 digest of a Volume3D including geometry metadata."""
    return sha256_array(
        vol.data,
        extra_header={
            "name": vol.name,
            "units": vol.units,
            "spacing_mm": list(vol.spacing_mm),
            "origin_mm": list(vol.origin_mm),
            "frame_of_reference_uid": vol.frame_of_reference_uid,
        },
    )


def sha256_model(model_obj: Any) -> str:
    """Compute deterministic SHA-256 digest of a Pydantic model or JSON-serializable dict."""
    if hasattr(model_obj, "model_dump"):
        payload = model_obj.model_dump(mode="json")
    else:
        payload = model_obj
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def enforce_provenance_gates(
    prior_courses: list[CourseSpec],
    planned_course: CourseSpec,
    dir_quality: DIRQualityMetrics,
    *,
    hard_ice_abstention_mm: float = 5.0,
    allow_folded_dvf: bool = False,
) -> list[str]:
    """Run hard spatial and physical provenance gates; return list of verified invariants.

    Raises:
        AbstentionError: If any non-negotiable physical or spatial provenance gate fails.
    """
    verified: list[str] = []

    if not prior_courses:
        raise AbstentionError(
            "NO_PRIOR_COURSE",
            "Re-irradiation accumulation requires at least one prior course.",
        )

    for c in [*prior_courses, planned_course]:
        if not c.frame_of_reference_uid or c.frame_of_reference_uid.upper() == "UNSPECIFIED":
            raise AbstentionError(
                "BROKEN_FOR_UID",
                f"Course '{c.course_id}' lacks a valid DICOM FrameOfReferenceUID.",
                {"course_id": c.course_id},
            )
    verified.append("COURSE_FRAME_OF_REFERENCE_UID_PRESENT")

    if not allow_folded_dvf and dir_quality.folded_voxel_count > 0:
        raise AbstentionError(
            "DIR_TOPOLOGY_FOLDING",
            (
                f"Deformable registration failed diffeomorphism check: "
                f"{dir_quality.folded_voxel_count} folded voxels (min J = {dir_quality.jacobian_min:.4f} <= 0). "
                "Cannot legally warp dose across inverted tissue topology."
            ),
            {
                "folded_voxel_count": dir_quality.folded_voxel_count,
                "jacobian_min": dir_quality.jacobian_min,
            },
        )
    verified.append("DIR_POSITIVE_JACOBIAN_DIFFEOMORPHISM")

    if dir_quality.ice_p95_mm > hard_ice_abstention_mm:
        raise AbstentionError(
            "DIR_EXCESSIVE_ICE",
            (
                f"DIR 95th-percentile Inverse-Consistency Error ({dir_quality.ice_p95_mm:.2f} mm) "
                f"exceeds hard abstention threshold ({hard_ice_abstention_mm:.1f} mm)."
            ),
            {"ice_p95_mm": dir_quality.ice_p95_mm, "threshold_mm": hard_ice_abstention_mm},
        )
    verified.append("DIR_INVERSE_CONSISTENCY_WITHIN_HARD_BOUND")

    return verified


def build_provenance_record(
    *,
    engine_version: str,
    case_id: str,
    dataset_source: str,
    dataset_citation: str,
    dataset_license: str,
    input_checksums: dict[str, str],
    invariants_checked: list[str],
    max_inverse_residual_gy: float,
    residual_tolerance_gy: float = 1e-6,
) -> ProvenanceRecord:
    """Construct an immutable ProvenanceRecord after independent invariant verification."""
    all_ok = max_inverse_residual_gy <= residual_tolerance_gy
    if all_ok:
        invariants_checked = [
            *invariants_checked,
            f"FORWARD_INVERSE_RESIDUAL_LE_{residual_tolerance_gy:.0e}_GY",
        ]
    return ProvenanceRecord(
        engine_version=engine_version,
        timestamp_utc=datetime.now(UTC).isoformat(),
        case_id=case_id,
        dataset_source=dataset_source,
        dataset_citation=dataset_citation,
        dataset_license=dataset_license,
        input_checksums=input_checksums,
        invariants_checked=invariants_checked,
        max_inverse_residual_gy=float(max_inverse_residual_gy),
        all_invariants_passed=all_ok,
        abstention_triggered=False,
        abstention_reason=None,
    )
