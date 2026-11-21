# Data Sources, Licensing, and Peer-Reviewed Citations

This repository adheres strictly to open-science attribution, copyright compliance, and
zero-plagiarism practices. Every public dataset, clinical parameter, and biophysical
equation used in `reirt-biobudget` is explicitly cited below.

---

## 1. Public Clinical Benchmark Dataset

### OpenKBP (Open-Access Knowledge-Based Planning Grand Challenge Dataset)
- **Files Used:** `data/openkbp/pt_1/`, `data/openkbp/pt_2/`, `data/openkbp/pt_3/` (de-identified head-and-neck IMRT 3D CT, clinical 3D dose distributions, voxel spacing, and contoured TG-263 OAR/PTV masks)
- **Authors:** Aaron Babier, Binghao Zhang, Rafid Mahmood, Kevin L. Moore, Thomas G. Purdie, Andrea L. McNiven, Timothy C. Y. Chan
- **Publication:** *Medical Physics*, 2021;48(9):5549–5561
- **DOI:** [`10.1002/mp.14845`](https://doi.org/10.1002/mp.14845)
- **Upstream Repository:** https://github.com/ababier/open-kbp
- **License:** **CC BY 4.0** (dataset) / **MIT License** (repository code; verbatim upstream license preserved at `data/openkbp/UPSTREAM_LICENSE_OPENKBP.txt`)
- **Attribution & Modification Disclosure:**
  - The Course-1 CT (`ct.csv`), Course-1 IMRT 3D dose (`dose.csv`), voxel dimensions, and contoured structures are used **verbatim** without altering upstream files, verified by SHA-256 checksums in `data/openkbp/pt_1/MANIFEST.json`.
  - Because OpenKBP provides single-course plans rather than multi-course clinical re-irradiation follow-ups, the Course-2 follow-up anatomy, analytical SVF diffeomorphic deformation, and candidate salvage Course-2 dose distribution are **explicitly generated at runtime** in `io_loaders.py` and tagged as derived longitudinal benchmark artifacts in the cryptographic provenance ledger.

```bibtex
@article{babier2021openkbp,
  title   = {OpenKBP: The open-access knowledge-based planning grand challenge and dataset},
  author  = {Babier, Aaron and Zhang, Binghao and Mahmood, Rafid and Moore, Kevin L. and Purdie, Thomas G. and McNiven, Andrea L. and Chan, Timothy C. Y.},
  journal = {Medical Physics},
  volume  = {48},
  number  = {9},
  pages   = {5549--5561},
  year    = {2021},
  doi     = {10.1002/mp.14845}
}
```

---

## 2. Clinical Consensus Guidelines & QA Standards

1. **ESTRO/EORTC Re-Irradiation Consensus (2022):**
   Andratschke N, Willmann J, Appelt AL, et al. *European Society for Radiotherapy and Oncology and European Organisation for Research and Treatment of Cancer consensus on re-irradiation: definition, reporting, and clinical decision making.* The Lancet Oncology. 2022;23(10):e469–e478. DOI: [`10.1016/S1470-2045(22)00447-8`](https://doi.org/10.1016/S1470-2045(22)00447-8)
2. **AAPM Task Group 132 (Deformable Image Registration QA, 2017):**
   Brock KK, Mutic S, McNutt TR, Li H, Kessler ML. *Use of image registration and fusion algorithms and techniques in radiotherapy: Report of the AAPM Radiation Therapy Committee Task Group No. 132.* Medical Physics. 2017;44(7):e43–e76. DOI: [`10.1002/mp.12256`](https://doi.org/10.1002/mp.12256)
3. **AAPM Task Group 263 (Standardized OAR Nomenclature, 2018):**
   Mayo CS, Moran JM, Bosch W, et al. *American Association of Physicists in Medicine Task Group 263: Standardizing Nomenclatures in Radiation Oncology.* Int J Radiat Oncol Biol Phys. 2018;100(4):1057–1066. DOI: [`10.1016/j.ijrobp.2017.12.013`](https://doi.org/10.1016/j.ijrobp.2017.12.013)

---

## 3. Radiobiology Models & Organ Tolerance Literature

4. **Linear-Quadratic (LQ) Model (1989):**
   Fowler JF. *The linear-quadratic formula and progress in fractionated radiotherapy.* Br J Radiol. 1989;62(740):679–694. DOI: [`10.1259/0007-1285-62-740-679`](https://doi.org/10.1259/0007-1285-62-740-679)
5. **Linear-Quadratic-Linear (LQL) Model for Hypofractionation (2008):**
   Astrahan MA. *Some implications of linear-quadratic-linear radiation dose-response with regard to hypofractionation.* Medical Physics. 2008;35(9):4161–4172. DOI: [`10.1118/1.2969065`](https://doi.org/10.1118/1.2969065)
6. **Lea–Catcheside Incomplete Repair Factor (1985):**
   Dale RG. *The application of the linear-quadratic dose-effect equation to fractionated and protracted radiotherapy.* Br J Radiol. 1985;58(690):515–528. DOI: [`10.1259/0007-1285-58-690-515`](https://doi.org/10.1259/0007-1285-58-690-515)
7. **Phenomenological Variable Proton RBE Model (2015):**
   McNamara AL, Schuemann J, Paganetti H. *A phenomenological relative biological effectiveness (RBE) model for proton therapy based on all published in vitro cell survival data.* Phys Med Biol. 2015;60(21):8399–8416. DOI: [`10.1088/0031-9155/60/21/8399`](https://doi.org/10.1088/0031-9155/60/21/8399)
8. **QUANTEC Organ Tolerances (2010):**
   Marks LB, Yorke ED, Jackson A, et al. *Use of normal tissue complication probability models in the clinic.* Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S10–S19. DOI: [`10.1016/j.ijrobp.2009.07.1754`](https://doi.org/10.1016/j.ijrobp.2009.07.1754)
9. **Spinal Cord Re-Irradiation Tolerance & Recovery (2005 / 2006):**
   - Nieder C, Grosu AL, Andratschke NH, Molls M. *Proposal of human spinal cord reirradiation dose based on collection of data from 40 patients.* Int J Radiat Oncol Biol Phys. 2005;61(3):851–855. DOI: [`10.1016/j.ijrobp.2004.06.016`](https://doi.org/10.1016/j.ijrobp.2004.06.016)
   - Nieder C, Grosu AL, Andratschke NH, Molls M. *Update of human spinal cord reirradiation tolerance based on additional data from 38 patients.* Int J Radiat Oncol Biol Phys. 2006;66(5):1446–1449. DOI: [`10.1016/j.ijrobp.2006.07.1383`](https://doi.org/10.1016/j.ijrobp.2006.07.1383)
10. **HyTEC Spinal Cord SBRT & Re-Irradiation Tolerance (2021):**
    Sahgal A, Chang JH, Ma L, et al. *Spinal Cord Dose Tolerance to Stereotactic Body Radiation Therapy.* Int J Radiat Oncol Biol Phys. 2021;110(1):124–136. DOI: [`10.1016/j.ijrobp.2019.09.038`](https://doi.org/10.1016/j.ijrobp.2019.09.038)

---

## 4. Normal Tissue Complication Probability (LKB / gEUD) & Global Sobol' Sensitivity

11. **Lyman NTCP Model (1985):**
    Lyman JT. *Complication probability as assessed from dose-volume histograms.* Radiat Res Suppl. 1985;8:S13–S19. DOI: [`10.2307/3583506`](https://doi.org/10.2307/3583506)
12. **Burman–Kutcher–Emami LKB Parameter Fits (1991):**
    Burman C, Kutcher GJ, Emami B, Goitein M. *Fitting of normal tissue tolerance data to an analytic function.* Int J Radiat Oncol Biol Phys. 1991;21(1):123–135. DOI: [`10.1016/0360-3016(91)90172-Z`](https://doi.org/10.1016/0360-3016(91)90172-Z)
13. **QUANTEC Spinal Cord Dose–Volume Effects (2010):**
    Kirkpatrick JP, van der Kogel AJ, Schultheiss TE. *Radiation dose-volume effects in the spinal cord.* Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S42–S49. DOI: [`10.1016/j.ijrobp.2009.04.095`](https://doi.org/10.1016/j.ijrobp.2009.04.095)
14. **QUANTEC Salivary Gland Dose–Volume Effects (2010):**
    Deasy JO, Moiseenko V, Marks L, Chao KSC, Nam J, Eisbruch A. *Radiotherapy dose-volume effects on salivary gland function.* Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S58–S63. DOI: [`10.1016/j.ijrobp.2009.06.090`](https://doi.org/10.1016/j.ijrobp.2009.06.090)
15. **Saltelli–Jansen Global Variance-Based Sensitivity Analysis (2010):**
    Saltelli A, Annoni P, Azzini I, Campolongo F, Ratto M, Tarantola S. *Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index.* Comput Phys Commun. 2010;181(2):259–270. DOI: [`10.1016/j.cpc.2009.09.018`](https://doi.org/10.1016/j.cpc.2009.09.018)
