"""Tests for multi-patient OpenKBP cohort benchmark, landmark TRE validation, and Fig 6-7 rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from reirt_biobudget.cohort import run_openkbp_cohort_benchmark
from reirt_biobudget.visualization import (
    plot_fig6_cohort_landmark_validation,
    plot_fig7_sobol_ntcp_analysis,
)

OPENKBP_ROOT = Path(__file__).resolve().parents[1] / "data" / "openkbp"
PATIENT_DIRS = [OPENKBP_ROOT / pid for pid in ("pt_1", "pt_2", "pt_3")]


@pytest.mark.skipif(
    not all(d.exists() for d in PATIENT_DIRS),
    reason="OpenKBP pt_1..pt_3 samples not all downloaded",
)
def test_openkbp_3patient_cohort_landmarks_k2_and_figures(tmp_path: Path) -> None:
    cohort = run_openkbp_cohort_benchmark(
        PATIENT_DIRS,
        tre_sigma_mm=2.5,
        n_ensemble=10,
        sobol_pow2=6,
        seed=42,
    )
    assert len(cohort.patients) == 3
    for p in cohort.patients:
        assert p.tg132_pass is True
        assert p.jacobian_min > 0.5
        assert p.ice_p95_mm < 0.05
        assert p.max_inverse_residual_gy < 1e-10
        lv = p.landmark_validation
        assert lv.n_landmarks >= 10
        assert lv.pre_dir_tre_mean_mm > 1.0
        assert lv.post_dir_tre_p95_mm < 0.05
        assert lv.empirical_90pct_envelope_coverage_pct >= 70.0
        assert len(p.ntcp_summaries) >= 3

    assert cohort.k2_multicourse_output is not None
    assert len(cohort.k2_multicourse_output.result.prior_courses) == 2
    assert cohort.k2_multicourse_output.result.provenance.max_inverse_residual_gy < 1e-10

    # Render Figure 6 & Figure 7 to tmp_path
    f6 = plot_fig6_cohort_landmark_validation(cohort, tmp_path)
    sobols = [p.sobol_spinal_cord for p in cohort.patients if p.sobol_spinal_cord is not None]
    f7 = plot_fig7_sobol_ntcp_analysis(sobols[:1], cohort.patients[0].ntcp_summaries, tmp_path)
    assert f6.exists() and f6.stat().st_size > 10_000
    assert f7.exists() and f7.stat().st_size > 10_000
