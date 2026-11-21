#!/usr/bin/env python3
"""Download a real, open-access de-identified head-and-neck clinical case from OpenKBP.

Source Dataset:
    OpenKBP: The open-access knowledge-based planning grand challenge and dataset
    Babier A, Zhang B, Mahmood R, Moore KL, Purdie TG, McNiven AL, Chan TCY.
    Medical Physics. 2021;48(9):5549-5561. DOI: 10.1002/mp.14845
    Repository: https://github.com/ababier/open-kbp
    Code License: MIT License; Dataset License: CC BY 4.0

This script downloads only the sparse CSV files for one or more de-identified patients
(default: pt_1) plus the upstream LICENSE file, records SHA-256 checksums, and writes
an explicit ATTRIBUTION.md notice alongside the downloaded files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/ababier/open-kbp/master"
OPENKBP_FILES = [
    "ct.csv",
    "dose.csv",
    "possible_dose_mask.csv",
    "voxel_dimensions.csv",
    "Brainstem.csv",
    "SpinalCord.csv",
    "RightParotid.csv",
    "LeftParotid.csv",
    "Esophagus.csv",
    "Larynx.csv",
    "Mandible.csv",
    "PTV70.csv",
    "PTV63.csv",
    "PTV56.csv",
]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fetch_url(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "reirt-biobudget/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def fetch_patient(patient_id: str, output_root: Path, split: str = "train-pats") -> dict:
    patient_dir = output_root / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: dict[str, dict[str, object]] = {}

    # Always save upstream LICENSE for transparent attribution
    license_url = f"{RAW_BASE}/LICENSE"
    license_bytes = _fetch_url(license_url)
    if license_bytes is not None:
        license_path = output_root / "UPSTREAM_LICENSE_OPENKBP.txt"
        license_path.write_bytes(license_bytes)

    for fname in OPENKBP_FILES:
        url = f"{RAW_BASE}/provided-data/{split}/{patient_id}/{fname}"
        payload = _fetch_url(url)
        if payload is None:
            continue
        dest = patient_dir / fname
        dest.write_bytes(payload)
        manifest_files[fname] = {
            "url": url,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }

    if "ct.csv" not in manifest_files or "dose.csv" not in manifest_files:
        raise RuntimeError(
            f"Failed to download mandatory OpenKBP files (ct.csv, dose.csv) for {patient_id}."
        )

    manifest = {
        "dataset": "OpenKBP (Open-Access Knowledge-Based Planning Grand Challenge Dataset)",
        "citation": (
            "Babier A, Zhang B, Mahmood R, Moore KL, Purdie TG, McNiven AL, Chan TCY. "
            "OpenKBP: The open-access knowledge-based planning grand challenge and dataset. "
            "Medical Physics. 2021;48(9):5549-5561. doi:10.1002/mp.14845"
        ),
        "repository": "https://github.com/ababier/open-kbp",
        "license": "MIT (code) / CC BY 4.0 (dataset)",
        "patient_id": patient_id,
        "split": split,
        "downloaded_utc": datetime.now(UTC).isoformat(),
        "files": manifest_files,
    }
    (patient_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    attribution_md = f"""# Data Attribution — OpenKBP (`{patient_id}`)

These files were retrieved verbatim from the public **OpenKBP** repository (`ababier/open-kbp`,
split `{split}`, patient `{patient_id}`) for non-commercial academic research and reproducible
benchmarking in `reirt-biobudget`.

- **Dataset:** OpenKBP: The open-access knowledge-based planning grand challenge and dataset
- **Authors:** Aaron Babier, Binghao Zhang, Rafid Mahmood, Kevin L. Moore, Thomas G. Purdie,
  Andrea L. McNiven, and Timothy C. Y. Chan
- **Publication:** *Medical Physics*, 2021;48(9):5549–5561
- **DOI:** `10.1002/mp.14845`
- **Source URL:** https://github.com/ababier/open-kbp
- **License:** CC BY 4.0 (dataset) / MIT (repository code; see `UPSTREAM_LICENSE_OPENKBP.txt`)
- **Underlying Clinical Provenance:** De-identified head-and-neck IMRT plans curated from public
  TCIA collections (Head-Neck-PET-CT, HNSCC, TCGA-HNSC, HNSCC-3DCT-RT).
- **Modifications:** None to raw downloaded CSVs (`ct.csv`, `dose.csv`, OAR masks,
  `voxel_dimensions.csv`). Longitudinal re-irradiation deformations used in benchmarks are
  explicitly generated at runtime from documented analytical diffeomorphisms and tagged as
  derived synthetic follow-up states in the provenance ledger.
"""
    (output_root / "ATTRIBUTION.md").write_text(attribution_md, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a public OpenKBP patient case.")
    parser.add_argument("--patient", default="pt_1", help="OpenKBP patient ID (e.g., pt_1)")
    parser.add_argument("--split", default="train-pats", help="OpenKBP split folder")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "openkbp"),
        help="Destination directory",
    )
    args = parser.parse_args()
    manifest = fetch_patient(args.patient, Path(args.out), split=args.split)
    print(f"Downloaded {len(manifest['files'])} files for {args.patient} to {args.out}")


if __name__ == "__main__":
    main()
