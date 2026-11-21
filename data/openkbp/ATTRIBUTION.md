# Data Attribution — OpenKBP (`pt_1`, `pt_2`, `pt_3`)

These files were retrieved verbatim from the public **OpenKBP** repository (`ababier/open-kbp`,
split `train-pats`, patients `pt_1`, `pt_2`, and `pt_3`) for non-commercial academic research
and reproducible multi-patient benchmarking in `reirt-biobudget`.

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
  `voxel_dimensions.csv`), each SHA-256 verified in its per-patient `MANIFEST.json`.
  Longitudinal re-irradiation deformations used in benchmarks are explicitly generated at
  runtime from documented analytical SVF diffeomorphisms and tagged as derived synthetic
  follow-up states in the cryptographic provenance ledger.
