"""Model Context Protocol (MCP) JSON-RPC 2.0 Stdio Server for reirt-biobudget.

Exposes the proprietary biophysical re-irradiation engine to frontier agent harnesses
via strictly typed, provenance-gated tools. Every numeric result is computed by the
deterministic physics kernel and returned alongside verification residuals and citations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from reirt_biobudget import __version__
from reirt_biobudget.harness.agent_orchestrator import run_frontier_investigation
from reirt_biobudget.io_loaders import (
    build_reirradiation_benchmark_from_openkbp,
    load_openkbp_patient,
)
from reirt_biobudget.ir import AbstentionError, CourseSpec, Modality, RadiobiologyModel
from reirt_biobudget.ontology import TG263_OAR_LIBRARY, get_oar_constraint, normalize_tg263
from reirt_biobudget.radiobiology import (
    conservative_recovery_fraction,
    eqd2_to_physical,
    physical_to_bed,
    physical_to_eqd2,
)

MCP_PROTOCOL_VERSION = "2024-11-05"

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "biobudget_lookup_tg263_oar",
        "description": (
            "Normalize an OAR name to AAPM TG-263 and return sourced radiobiology parameters "
            "(alpha/beta, uncertainty range, LQL d_T, cumulative EQD2 limits, recovery ceiling, DOIs)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "oar_name": {"type": "string", "description": "Raw or TG-263 OAR name (e.g. 'cord', 'SpinalCord')."}
            },
            "required": ["oar_name"],
        },
    },
    {
        "name": "biobudget_compute_eqd2",
        "description": (
            "Exact deterministic forward radiobiology solver: physical dose (Gy) -> BED & EQD2 (Gy) "
            "across LQ, LQL (Astrahan 2008), Lea-Catcheside (Dale 1985), or McNamara variable proton RBE."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "physical_dose_gy": {"type": "number", "minimum": 0.0},
                "n_fractions": {"type": "integer", "minimum": 1},
                "alpha_beta_gy": {"type": "number", "exclusiveMinimum": 0.0},
                "model": {
                    "type": "string",
                    "enum": ["lq", "lql", "lea_catcheside", "mcnamara_proton"],
                    "default": "lq",
                },
                "modality": {
                    "type": "string",
                    "enum": ["photon", "proton_fixed_rbe", "proton_variable_rbe", "brachy_hdr", "brachy_ldr"],
                    "default": "photon",
                },
                "lql_d_t_gy": {"type": "number", "default": 6.0},
                "repair_half_time_h": {"type": "number", "default": 1.5},
                "fraction_duration_h": {"type": "number", "default": 0.0},
                "fractions_per_day": {"type": "integer", "default": 1},
                "interfraction_interval_h": {"type": "number", "default": 24.0},
                "let_d_kev_um": {"type": "number", "default": 2.5},
            },
            "required": ["physical_dose_gy", "n_fractions", "alpha_beta_gy"],
        },
    },
    {
        "name": "biobudget_invert_remaining_dose",
        "description": (
            "Exact analytical inverse solver: convert remaining OAR EQD2 budget (Gy) into "
            "maximum safe Course-2 physical dose (Gy) for a chosen schedule/modality, with "
            "independent forward-inverse residual verification."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "remaining_eqd2_gy": {"type": "number", "minimum": 0.0},
                "n_fractions": {"type": "integer", "minimum": 1},
                "alpha_beta_gy": {"type": "number", "exclusiveMinimum": 0.0},
                "model": {
                    "type": "string",
                    "enum": ["lq", "lql", "lea_catcheside", "mcnamara_proton"],
                    "default": "lql",
                },
                "modality": {
                    "type": "string",
                    "enum": ["photon", "proton_fixed_rbe", "proton_variable_rbe", "brachy_hdr", "brachy_ldr"],
                    "default": "photon",
                },
                "lql_d_t_gy": {"type": "number", "default": 6.0},
                "repair_half_time_h": {"type": "number", "default": 1.5},
                "fractions_per_day": {"type": "integer", "default": 1},
                "interfraction_interval_h": {"type": "number", "default": 24.0},
                "let_d_kev_um": {"type": "number", "default": 2.5},
            },
            "required": ["remaining_eqd2_gy", "n_fractions", "alpha_beta_gy"],
        },
    },
    {
        "name": "biobudget_compute_recovery_credit",
        "description": (
            "Compute conservative OAR time-dependent tissue recovery credit with hard zero-recovery "
            "enforcement below minimum elapsed months (Nieder 2006 / Sahgal HyTEC 2021)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "oar_name": {"type": "string"},
                "months_elapsed": {"type": "number", "minimum": 0.0},
                "override_fraction": {"type": ["number", "null"], "default": None},
            },
            "required": ["oar_name", "months_elapsed"],
        },
    },
    {
        "name": "biobudget_run_openkbp_investigation",
        "description": (
            "Run the complete 3D diffeomorphic DIR-uncertainty envelope, adversarial assumption "
            "tipping-point sweep, 6-schedule tradeoff matrix, and independent Critic-Verifier audit "
            "on a public OpenKBP clinical head-and-neck case."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "patient_dir": {"type": "string", "description": "Path to OpenKBP patient directory (e.g. data/openkbp/pt_1)."},
                "months_elapsed": {"type": "number", "default": 18.0},
                "planned_n_fractions": {"type": "integer", "default": 5},
                "planned_rx_gy": {"type": "number", "default": 35.0},
                "tre_sigma_mm": {"type": "number", "default": 2.5},
                "n_ensemble": {"type": "integer", "default": 16},
            },
            "required": ["patient_dir"],
        },
    },
]


def _make_course(args: dict[str, Any]) -> CourseSpec:
    return CourseSpec(
        course_id="MCP_COURSE",
        label="MCP Evaluated Course",
        modality=Modality(args.get("modality", "photon")),
        model=RadiobiologyModel(args.get("model", "lq")),
        n_fractions=int(args["n_fractions"]),
        prescription_gy=max(float(args.get("physical_dose_gy", 30.0)), 1.0),
        frame_of_reference_uid="1.2.826.0.1.3680043.8.498.MCP.SCALAR",
        d_t_gy=float(args.get("lql_d_t_gy", 6.0)),
        fraction_duration_h=float(args.get("fraction_duration_h", 0.0)),
        fractions_per_day=int(args.get("fractions_per_day", 1)),
        interfraction_interval_h=float(args.get("interfraction_interval_h", 24.0)),
        mean_let_d_kev_um=float(args.get("let_d_kev_um", 2.5)),
    )


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a registered MCP tool deterministically and return structured JSON payload."""
    if name == "biobudget_lookup_tg263_oar":
        raw = str(arguments["oar_name"])
        canonical = normalize_tg263(raw)
        cons = get_oar_constraint(canonical)
        if cons is None:
            return {
                "found": False,
                "normalized_name": canonical,
                "available_tg263_oars": sorted(TG263_OAR_LIBRARY.keys()),
            }
        return {"found": True, "constraint": cons.model_dump(mode="json")}

    if name == "biobudget_compute_eqd2":
        course = _make_course(arguments)
        ab = float(arguments["alpha_beta_gy"])
        dose = float(arguments["physical_dose_gy"])
        t_half = float(arguments.get("repair_half_time_h", 1.5))
        let_d = float(arguments.get("let_d_kev_um", 2.5))
        bed = float(
            physical_to_bed(dose, course, ab, repair_half_time_h=t_half, let_d_kev_um=let_d)
        )
        eqd2 = float(
            physical_to_eqd2(dose, course, ab, repair_half_time_h=t_half, let_d_kev_um=let_d)
        )
        return {
            "physical_dose_gy": dose,
            "dose_per_fraction_gy": dose / course.n_fractions,
            "bed_gy": bed,
            "eqd2_gy": eqd2,
            "model": course.model.value,
            "modality": course.modality.value,
            "alpha_beta_gy": ab,
        }

    if name == "biobudget_invert_remaining_dose":
        course = _make_course({**arguments, "model": arguments.get("model", "lql")})
        ab = float(arguments["alpha_beta_gy"])
        rem_eqd2 = float(arguments["remaining_eqd2_gy"])
        t_half = float(arguments.get("repair_half_time_h", 1.5))
        let_d = float(arguments.get("let_d_kev_um", 2.5))
        rem_phys = float(
            eqd2_to_physical(rem_eqd2, course, ab, repair_half_time_h=t_half, let_d_kev_um=let_d)
        )
        recomp = float(
            physical_to_eqd2(rem_phys, course, ab, repair_half_time_h=t_half, let_d_kev_um=let_d)
        )
        residual = abs(recomp - rem_eqd2)
        return {
            "remaining_eqd2_gy": rem_eqd2,
            "remaining_physical_dose_gy": rem_phys,
            "dose_per_fraction_ceiling_gy": rem_phys / course.n_fractions,
            "verified_forward_inverse_residual_gy": residual,
            "invariant_verified": residual <= 1e-8,
            "model": course.model.value,
            "modality": course.modality.value,
        }

    if name == "biobudget_compute_recovery_credit":
        cons = get_oar_constraint(str(arguments["oar_name"]))
        if cons is None:
            raise ValueError(f"Unknown TG-263 OAR '{arguments['oar_name']}'")
        r = conservative_recovery_fraction(
            cons,
            months_elapsed=float(arguments["months_elapsed"]),
            override_fraction=arguments.get("override_fraction"),
        )
        return {
            "tg263_name": cons.tg263_name,
            "months_elapsed": float(arguments["months_elapsed"]),
            "recovery_fraction": r,
            "effective_prior_multiplier": 1.0 - r,
            "max_conservative_recovery_fraction": cons.max_recovery_fraction,
            "min_recovery_months": cons.min_recovery_months,
            "citation": cons.citation,
            "doi": cons.doi,
        }

    if name == "biobudget_run_openkbp_investigation":
        patient = load_openkbp_patient(Path(arguments["patient_dir"]))
        case = build_reirradiation_benchmark_from_openkbp(
            patient,
            months_elapsed=float(arguments.get("months_elapsed", 18.0)),
            planned_n_fractions=int(arguments.get("planned_n_fractions", 5)),
            planned_rx_gy=float(arguments.get("planned_rx_gy", 35.0)),
        )
        report = run_frontier_investigation(
            case,
            tre_sigma_mm=float(arguments.get("tre_sigma_mm", 2.5)),
            n_ensemble=int(arguments.get("n_ensemble", 16)),
        )
        return {
            "baseline_result": report.baseline.result.model_dump(mode="json"),
            "tipping_points": [tp.model_dump(mode="json") for tp in report.tipping_points],
            "schedule_matrix": [row.model_dump(mode="json") for row in report.schedule_matrix],
            "critic_audit": report.critic_audit.model_dump(mode="json"),
        }

    raise KeyError(f"Unknown MCP tool '{name}'")


def handle_jsonrpc_message(request: dict[str, Any]) -> dict[str, Any] | None:
    """Process one MCP JSON-RPC 2.0 request and return the response dict (or None for notifications)."""
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params") or {}

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "reirt-biobudget-mcp", "version": __version__},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOL_SCHEMAS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments") or {}
        try:
            output = dispatch_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(output, indent=2)}],
                    "isError": False,
                },
            }
        except AbstentionError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "status": "ABSTAIN",
                                    "code": exc.code,
                                    "message": str(exc),
                                    "details": exc.details,
                                },
                                indent=2,
                            ),
                        }
                    ],
                    "isError": True,
                },
            }
        except (ValueError, KeyError, FileNotFoundError, TypeError, RuntimeError) as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Tool error: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def serve_stdio() -> None:
    """Run the MCP JSON-RPC 2.0 server over stdin/stdout line-delimited JSON."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_jsonrpc_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
