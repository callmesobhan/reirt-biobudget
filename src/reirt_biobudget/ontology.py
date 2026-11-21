"""AAPM TG-263 OAR Nomenclature Normalizer & Sourced Clinical Re-Irradiation Constraints.

Every biological parameter (alpha/beta, uncertainty bounds, repair half-time, LQL
transition dose d_T, cumulative EQD2 tolerances, and conservative tissue recovery ceilings)
is explicitly tied to peer-reviewed literature with DOIs.

References:
    - Mayo CS et al. AAPM Task Group 263: Standardizing Nomenclatures in Radiation
      Oncology. Int J Radiat Oncol Biol Phys. 2018;100(4):1057-1066.
      DOI: 10.1016/j.ijrobp.2017.12.013
    - Andratschke N et al. ESTRO/EORTC consensus on re-irradiation. Lancet Oncol.
      2022;23(10):e469-e478. DOI: 10.1016/S1470-2045(22)00447-8
    - Nieder C et al. Update of human spinal cord reirradiation tolerance. Int J Radiat
      Oncol Biol Phys. 2006;66(5):1446-1449. DOI: 10.1016/j.ijrobp.2006.07.1383
    - Sahgal A et al. HyTEC Spinal Cord Dose Tolerance to SBRT. Int J Radiat Oncol Biol
      Phys. 2021;110(1):124-136. DOI: 10.1016/j.ijrobp.2019.09.038
    - Marks LB et al. Use of normal tissue complication probability models in the clinic
      (QUANTEC). Int J Radiat Oncol Biol Phys. 2010;76(3 Suppl):S10-S19.
      DOI: 10.1016/j.ijrobp.2009.07.1754
"""

from __future__ import annotations

import re

from reirt_biobudget.ir import OARConstraint

# Canonical TG-263 -> clinical radiobiology & cumulative re-irradiation constraints.
# Values represent conservative research defaults; clinical decisions require institutional
# peer review and patient-specific physician approval.
TG263_OAR_LIBRARY: dict[str, OARConstraint] = {
    "SpinalCord": OARConstraint(
        tg263_name="SpinalCord",
        organ_class="serial",
        alpha_beta_gy=2.0,
        alpha_beta_low_gy=1.5,
        alpha_beta_high_gy=3.0,
        repair_half_time_h=2.0,
        lql_d_t_gy=6.0,
        primary_metric="D0.03cc",
        eqd2_limit_gy=60.0,
        eqd2_caution_gy=50.0,
        max_recovery_fraction=0.25,
        min_recovery_months=6.0,
        citation=(
            "Nieder et al. (IJROBP 2006;66:1446-1449); Sahgal et al. HyTEC "
            "(IJROBP 2021;110:124-136); Andratschke et al. ESTRO/EORTC (Lancet Oncol 2022)"
        ),
        doi="10.1016/j.ijrobp.2006.07.1383",
    ),
    "Brainstem": OARConstraint(
        tg263_name="Brainstem",
        organ_class="serial",
        alpha_beta_gy=2.0,
        alpha_beta_low_gy=1.5,
        alpha_beta_high_gy=3.0,
        repair_half_time_h=2.0,
        lql_d_t_gy=6.0,
        primary_metric="D0.03cc",
        eqd2_limit_gy=64.0,
        eqd2_caution_gy=54.0,
        max_recovery_fraction=0.25,
        min_recovery_months=6.0,
        citation=(
            "Marks et al. QUANTEC (IJROBP 2010;76:S10-S19); "
            "Andratschke et al. ESTRO/EORTC (Lancet Oncol 2022;23:e469-e478)"
        ),
        doi="10.1016/j.ijrobp.2009.07.1754",
    ),
    "Bone_Mandible": OARConstraint(
        tg263_name="Bone_Mandible",
        organ_class="serial",
        alpha_beta_gy=3.0,
        alpha_beta_low_gy=2.0,
        alpha_beta_high_gy=4.0,
        repair_half_time_h=1.5,
        lql_d_t_gy=7.0,
        primary_metric="D0.03cc",
        eqd2_limit_gy=85.0,
        eqd2_caution_gy=70.0,
        max_recovery_fraction=0.10,
        min_recovery_months=12.0,
        citation=(
            "Marks et al. QUANTEC (IJROBP 2010); "
            "Andratschke et al. ESTRO/EORTC (Lancet Oncol 2022)"
        ),
        doi="10.1016/S1470-2045(22)00447-8",
    ),
    "Parotid_L": OARConstraint(
        tg263_name="Parotid_L",
        organ_class="parallel",
        alpha_beta_gy=3.0,
        alpha_beta_low_gy=2.0,
        alpha_beta_high_gy=4.5,
        repair_half_time_h=1.5,
        lql_d_t_gy=7.0,
        primary_metric="Dmean",
        eqd2_limit_gy=35.0,
        eqd2_caution_gy=26.0,
        max_recovery_fraction=0.15,
        min_recovery_months=12.0,
        citation="Marks et al. QUANTEC (IJROBP 2010;76:S10-S19)",
        doi="10.1016/j.ijrobp.2009.07.1754",
    ),
    "Parotid_R": OARConstraint(
        tg263_name="Parotid_R",
        organ_class="parallel",
        alpha_beta_gy=3.0,
        alpha_beta_low_gy=2.0,
        alpha_beta_high_gy=4.5,
        repair_half_time_h=1.5,
        lql_d_t_gy=7.0,
        primary_metric="Dmean",
        eqd2_limit_gy=35.0,
        eqd2_caution_gy=26.0,
        max_recovery_fraction=0.15,
        min_recovery_months=12.0,
        citation="Marks et al. QUANTEC (IJROBP 2010;76:S10-S19)",
        doi="10.1016/j.ijrobp.2009.07.1754",
    ),
    "Larynx": OARConstraint(
        tg263_name="Larynx",
        organ_class="mixed",
        alpha_beta_gy=3.0,
        alpha_beta_low_gy=2.0,
        alpha_beta_high_gy=4.0,
        repair_half_time_h=1.5,
        lql_d_t_gy=7.0,
        primary_metric="Dmean",
        eqd2_limit_gy=55.0,
        eqd2_caution_gy=44.0,
        max_recovery_fraction=0.10,
        min_recovery_months=12.0,
        citation="Marks et al. QUANTEC (IJROBP 2010;76:S10-S19)",
        doi="10.1016/j.ijrobp.2009.07.1754",
    ),
    "Esophagus": OARConstraint(
        tg263_name="Esophagus",
        organ_class="mixed",
        alpha_beta_gy=3.0,
        alpha_beta_low_gy=2.5,
        alpha_beta_high_gy=4.0,
        repair_half_time_h=1.5,
        lql_d_t_gy=7.0,
        primary_metric="D0.03cc",
        eqd2_limit_gy=75.0,
        eqd2_caution_gy=65.0,
        max_recovery_fraction=0.10,
        min_recovery_months=12.0,
        citation=(
            "Marks et al. QUANTEC (IJROBP 2010); "
            "Andratschke et al. ESTRO/EORTC (Lancet Oncol 2022)"
        ),
        doi="10.1016/S1470-2045(22)00447-8",
    ),
    "OpticChiasm": OARConstraint(
        tg263_name="OpticChiasm",
        organ_class="serial",
        alpha_beta_gy=2.0,
        alpha_beta_low_gy=1.5,
        alpha_beta_high_gy=3.0,
        repair_half_time_h=2.0,
        lql_d_t_gy=6.0,
        primary_metric="D0.03cc",
        eqd2_limit_gy=60.0,
        eqd2_caution_gy=54.0,
        max_recovery_fraction=0.15,
        min_recovery_months=12.0,
        citation="Marks et al. QUANTEC (IJROBP 2010); Andratschke et al. (Lancet Oncol 2022)",
        doi="10.1016/j.ijrobp.2009.07.1754",
    ),
    "Rectum": OARConstraint(
        tg263_name="Rectum",
        organ_class="mixed",
        alpha_beta_gy=3.0,
        alpha_beta_low_gy=2.5,
        alpha_beta_high_gy=4.0,
        repair_half_time_h=1.5,
        lql_d_t_gy=7.0,
        primary_metric="D0.03cc",
        eqd2_limit_gy=85.0,
        eqd2_caution_gy=75.0,
        max_recovery_fraction=0.10,
        min_recovery_months=12.0,
        citation="Marks et al. QUANTEC (IJROBP 2010); Andratschke et al. (Lancet Oncol 2022)",
        doi="10.1016/j.ijrobp.2009.07.1754",
    ),
}

# Alias map covering OpenKBP, Eclipse, RayStation, Pinnacle, Monaco, and institutional names.
_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(spinal[_\s-]?cord|cord|sc|spinalcanal)$", re.IGNORECASE), "SpinalCord"),
    (re.compile(r"^(brain[_\s-]?stem|bs)$", re.IGNORECASE), "Brainstem"),
    (re.compile(r"^(bone[_\s-]?)?mandible$", re.IGNORECASE), "Bone_Mandible"),
    (re.compile(r"^(left[_\s-]?parotid|parotid[_\s-]?l|l[_\s-]?parotid|parotid[_\s-]?left)$", re.IGNORECASE), "Parotid_L"),
    (re.compile(r"^(right[_\s-]?parotid|parotid[_\s-]?r|r[_\s-]?parotid|parotid[_\s-]?right)$", re.IGNORECASE), "Parotid_R"),
    (re.compile(r"^(larynx|glottis)$", re.IGNORECASE), "Larynx"),
    (re.compile(r"^(esophagus|oesophagus|esoph)$", re.IGNORECASE), "Esophagus"),
    (re.compile(r"^(optic[_\s-]?chiasm|chiasm)$", re.IGNORECASE), "OpticChiasm"),
    (re.compile(r"^(rectum|anorectum)$", re.IGNORECASE), "Rectum"),
]


def normalize_tg263(raw_name: str) -> str:
    """Normalize an institutional or dataset structure name to canonical AAPM TG-263."""
    cleaned = raw_name.strip()
    if cleaned in TG263_OAR_LIBRARY:
        return cleaned
    for pattern, canonical in _ALIAS_PATTERNS:
        if pattern.match(cleaned):
            return canonical
    return cleaned


def get_oar_constraint(raw_name: str) -> OARConstraint | None:
    """Return the sourced TG-263 OARConstraint for a raw or canonical structure name."""
    canonical = normalize_tg263(raw_name)
    return TG263_OAR_LIBRARY.get(canonical)
