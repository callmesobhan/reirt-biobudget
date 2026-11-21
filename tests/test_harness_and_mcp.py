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
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert "biobudget_compute_eqd2" in tool_names
    assert "biobudget_invert_remaining_dose" in tool_names
    assert "biobudget_lookup_tg263_oar" in tool_names

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
