"""Tests for Frontier Agent Harness (adversarial sweep + critic audit) and MCP JSON-RPC 2.0 server."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reirt_biobudget.harness.agent_orchestrator import (
    emit_estro_eortc_markdown,
    run_frontier_investigation,
)
from reirt_biobudget.harness.mcp_server import handle_jsonrpc_message
from reirt_biobudget.io_loaders import (
    build_reirradiation_benchmark_from_openkbp,
    load_openkbp_patient,
)

PT1_DIR = Path(__file__).resolve().parents[1] / "data" / "openkbp" / "pt_1"


def test_mcp_protocol_initialize_list_and_exact_physics_tools() -> None:
    init_resp = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "reirt-biobudget-mcp"

    list_resp = handle_jsonrpc_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert list_resp is not None
    tools = list_resp["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert tool_names == {
        "biobudget_lookup_tg263_oar",
        "biobudget_compute_eqd2",
        "biobudget_invert_remaining_dose",
        "biobudget_compute_recovery_credit",
        "biobudget_run_openkbp_investigation",
        "biobudget_compute_lkb_ntcp",
        "biobudget_run_sobol_sensitivity",
        "biobudget_parse_legacy_summary",
        "biobudget_reconstruct_legacy_budget",
    }
    for tool in tools:
        assert isinstance(tool["description"], str) and len(tool["description"]) > 20
        ann = tool["annotations"]
        assert ann == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }

    # Lookup messy alias -> TG-263
    lookup_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "biobudget_lookup_tg263_oar", "arguments": {"oar_name": "spinal_cord"}},
        }
    )
    assert lookup_resp is not None
    payload = json.loads(lookup_resp["result"]["content"][0]["text"])
    assert payload["found"] is True
    assert payload["constraint"]["tg263_name"] == "SpinalCord"
    assert payload["constraint"]["alpha_beta_gy"] == 2.0

    # Invert remaining 25 Gy EQD2 into 5-fx LQL physical dose ceiling and verify < 1e-8 Gy residual
    inv_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "biobudget_invert_remaining_dose",
                "arguments": {
                    "remaining_eqd2_gy": 25.0,
                    "n_fractions": 5,
                    "alpha_beta_gy": 2.0,
                    "model": "lql",
                    "lql_d_t_gy": 6.0,
                },
            },
        }
    )
    assert inv_resp is not None
    inv_data = json.loads(inv_resp["result"]["content"][0]["text"])
    assert inv_data["invariant_verified"] is True
    assert inv_data["verified_forward_inverse_residual_gy"] < 1e-10
    assert inv_data["remaining_physical_dose_gy"] > 0.0

    # Cross-check with forward tool
    fwd_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "biobudget_compute_eqd2",
                "arguments": {
                    "physical_dose_gy": inv_data["remaining_physical_dose_gy"],
                    "n_fractions": 5,
                    "alpha_beta_gy": 2.0,
                    "model": "lql",
                    "lql_d_t_gy": 6.0,
                },
            },
        }
    )
    assert fwd_resp is not None
    fwd_data = json.loads(fwd_resp["result"]["content"][0]["text"])
    assert math.isclose(fwd_data["eqd2_gy"], 25.0, abs_tol=1e-10)

    # Recovery credit tool
    rec_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "biobudget_compute_recovery_credit",
                "arguments": {"oar_name": "SpinalCord", "months_elapsed": 18.0},
            },
        }
    )
    assert rec_resp is not None and rec_resp["result"]["isError"] is False
    rec_data = json.loads(rec_resp["result"]["content"][0]["text"])
    assert rec_data["tg263_name"] == "SpinalCord"
    assert 0.0 <= rec_data["recovery_fraction"] <= 0.25

    # LKB NTCP tool
    ntcp_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "biobudget_compute_lkb_ntcp",
                "arguments": {"oar_name": "SpinalCord", "eqd2_voxels_gy": [45.0, 50.0, 52.0]},
            },
        }
    )
    assert ntcp_resp is not None and ntcp_resp["result"]["isError"] is False
    ntcp_data = json.loads(ntcp_resp["result"]["content"][0]["text"])
    assert ntcp_data["tg263_name"] == "SpinalCord"
    assert 0.0 <= ntcp_data["ntcp_pct"] <= 100.0

    # Parse legacy summary tool
    legacy_text = (
        "Prior Course 1 (2024): IMRT Photons, 70 Gy in 35 fractions (2.0 Gy/fx).\n"
        "SpinalCord Dmax = 42.0 Gy; Brainstem Dmax = 48.0 Gy."
    )
    leg_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "biobudget_parse_legacy_summary",
                "arguments": {"summary_text": legacy_text, "record_id": "MCP_LEG_01"},
            },
        }
    )
    assert leg_resp is not None and leg_resp["result"]["isError"] is False
    leg_data = json.loads(leg_resp["result"]["content"][0]["text"])
    assert leg_data["n_fractions"] == 35


@pytest.mark.skipif(not PT1_DIR.exists(), reason="OpenKBP pt_1 sample not downloaded")
def test_mcp_openkbp_investigation_sobol_and_legacy_reconstruct_tools() -> None:
    inv_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "biobudget_run_openkbp_investigation",
                "arguments": {"patient_dir": str(PT1_DIR), "n_ensemble": 6},
            },
        }
    )
    assert inv_resp is not None and inv_resp["result"]["isError"] is False
    inv_data = json.loads(inv_resp["result"]["content"][0]["text"])
    assert inv_data["critic_audit"]["passed"] is True

    sob_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "biobudget_run_sobol_sensitivity",
                "arguments": {
                    "patient_dir": str(PT1_DIR),
                    "oar_name": "SpinalCord",
                    "n_base_pow2": 4,
                },
            },
        }
    )
    assert sob_resp is not None and sob_resp["result"]["isError"] is False
    sob_data = json.loads(sob_resp["result"]["content"][0]["text"])
    assert sob_data["oar_name"] == "SpinalCord"

    recon_resp = handle_jsonrpc_message(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "biobudget_reconstruct_legacy_budget",
                "arguments": {"patient_dir": str(PT1_DIR), "planned_n_fractions": 5},
            },
        }
    )
    assert recon_resp is not None and recon_resp["result"]["isError"] is False
    recon_data = json.loads(recon_resp["result"]["content"][0]["text"])
    assert recon_data["max_inverse_residual_gy"] < 1e-10
    assert recon_data["truth_voxel_upper_safety_rate_pct"] >= 90.0


@pytest.mark.skipif(not PT1_DIR.exists(), reason="OpenKBP pt_1 sample not downloaded")
def test_frontier_harness_adversarial_sweep_and_critic_audit() -> None:
    patient = load_openkbp_patient(PT1_DIR)
    case = build_reirradiation_benchmark_from_openkbp(patient)
    report = run_frontier_investigation(case, tre_sigma_mm=2.5, n_ensemble=8, seed=42)

    assert report.critic_audit.passed is True
    assert report.critic_audit.max_recomputed_discrepancy_gy < 1e-8
    assert len(report.tipping_points) >= 4
    assert len(report.schedule_matrix) == 6
    assert report.phase_diagram.cum_eqd2_p95_matrix.shape == (5, 15)

    md = emit_estro_eortc_markdown(report)
    assert "ESTRO/EORTC Re-Irradiation Biophysical Dossier" in md
    assert "10.1002/mp.14845" in md
    assert "NOT A MEDICAL DEVICE" in md
