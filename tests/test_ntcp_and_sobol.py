"""Tests for LKB / gEUD cumulative NTCP, Saltelli-Sobol global sensitivity, and new MCP tools."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from reirt_biobudget.harness.mcp_server import handle_jsonrpc_message
from reirt_biobudget.ir import CourseSpec, Modality, RadiobiologyModel
from reirt_biobudget.ntcp_sobol import (
    compute_geud,
    compute_lkb_ntcp_from_geud,
    evaluate_oar_ntcp_envelope,
    get_lkb_spec,
    run_joint_sobol_sensitivity,
)

PT1_DIR = Path(__file__).resolve().parents[1] / "data" / "openkbp" / "pt_1"


def test_geud_and_lkb_ntcp_closed_form_and_overflow_safety() -> None:
    cord = get_lkb_spec("SpinalCord")
    assert cord is not None

    # Uniform dose equal to TD50 must yield gEUD == TD50 and NTCP == 50.0%
    uniform = np.full(500, cord.td50_gy, dtype=np.float64)
    geud = compute_geud(uniform, cord.n)
    assert math.isclose(geud, cord.td50_gy, abs_tol=1e-10)
    ntcp = float(compute_lkb_ntcp_from_geud(geud, cord.td50_gy, cord.m))
    assert math.isclose(ntcp, 0.5, abs_tol=1e-12)

    # Overflow safety at high dose (120 Gy) and very small serial n = 0.05 (1/n = 20)
    extreme = np.linspace(10.0, 120.0, 200)
    geud_ext = compute_geud(extreme, n=0.05)
    assert math.isfinite(geud_ext)
    assert 100.0 < geud_ext <= 120.0

    # Envelope ordering invariant
    env = {
        "cum_p05": np.full(100, 42.0),
        "cum_nominal": np.full(100, 50.0),
        "cum_p95": np.full(100, 58.0),
        "cum_worst": np.full(100, 62.0),
    }
    summary = evaluate_oar_ntcp_envelope("SpinalCord", env)
    assert summary is not None
    assert summary.ntcp_p05_pct <= summary.ntcp_nominal_pct <= summary.ntcp_p95_pct <= summary.ntcp_worst_pct


def test_saltelli_sobol_bounds_and_dominant_driver() -> None:
    rng = np.random.default_rng(11)
    # Build synthetic DIR ensemble with large spatial spread (dominant driver)
    ens = np.vstack([np.full(50, 25.0 + 18.0 * u) for u in np.linspace(0.0, 1.0, 16)])
    c2 = np.full(50, 10.0) + rng.normal(0.0, 0.2, size=50)
    c1_spec = CourseSpec(
        course_id="C1",
        label="Prior",
        n_fractions=35,
        prescription_gy=70.0,
        months_before_reirt=18.0,
        frame_of_reference_uid="1.2.3.4.S1",
    )
    c2_spec = CourseSpec(
        course_id="C2",
        label="Planned",
        modality=Modality.PHOTON,
        model=RadiobiologyModel.LQL,
        n_fractions=5,
        prescription_gy=35.0,
        frame_of_reference_uid="1.2.3.4.S2",
    )

    res = run_joint_sobol_sensitivity(
        oar_name="SpinalCord",
        prior_ensemble_vox=ens,
        candidate_course2_vox=c2,
        prior_course=c1_spec,
        planned_course=c2_spec,
        voxel_vol_cc=0.04,
        output_mode="eqd2",
        n_base_pow2=7,
        seed=42,
    )
    assert res.total_evaluations == 128 * 6
    assert res.dominant_parameter == "DIR Spatial Registration"
    for p in res.parameter_names:
        assert 0.0 <= res.first_order_s1[p] <= 1.0
        assert 0.0 <= res.total_order_st[p] <= 1.0
        assert res.interaction_st_minus_s1[p] >= 0.0
    assert res.output_p05 <= res.output_p50 <= res.output_p95


def test_mcp_lkb_ntcp_and_sobol_tools() -> None:
    # 1. LKB NTCP MCP tool
    resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "biobudget_compute_lkb_ntcp",
                "arguments": {"oar_name": "spinal_cord", "eqd2_voxels_gy": [66.5, 66.5, 66.5]},
            },
        }
    )
    assert resp is not None
    assert resp["result"]["isError"] is False
    data = json.loads(resp["result"]["content"][0]["text"])
    assert data["tg263_name"] == "SpinalCord"
    assert math.isclose(data["ntcp_pct"], 50.0, abs_tol=1e-2)

    # 2. Sobol MCP tool on real OpenKBP pt_1
    if PT1_DIR.exists():
        s_resp = handle_jsonrpc_message(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "biobudget_run_sobol_sensitivity",
                    "arguments": {
                        "patient_dir": str(PT1_DIR),
                        "oar_name": "SpinalCord",
                        "output_mode": "eqd2",
                        "n_base_pow2": 6,
                    },
                },
            }
        )
        assert s_resp is not None
        assert s_resp["result"]["isError"] is False
        s_data = json.loads(s_resp["result"]["content"][0]["text"])
        assert s_data["oar_name"] == "SpinalCord"
        assert s_data["dominant_parameter"] in s_data["parameter_names"]
