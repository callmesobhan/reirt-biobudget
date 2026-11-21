"""Command-Line Interface for reirt-biobudget."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reirt_biobudget_mpl")

import argparse
import json
from pathlib import Path

from reirt_biobudget.harness.agent_orchestrator import (
    emit_estro_eortc_markdown,
    run_frontier_investigation,
)
from reirt_biobudget.harness.mcp_server import serve_stdio
from reirt_biobudget.io_loaders import (
    build_reirradiation_benchmark_from_openkbp,
    load_openkbp_patient,
)
from reirt_biobudget.ontology import get_oar_constraint, normalize_tg263
from reirt_biobudget.visualization import generate_all_publication_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reirt-biobudget",
        description="3D Deformable Biophysical Dose-Accumulation & Remaining-Budget Engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run full re-irradiation investigation & generate figures/dossier.")
    p_run.add_argument("--patient-dir", default="data/openkbp/pt_1", help="Path to OpenKBP patient directory.")
    p_run.add_argument("--months-elapsed", type=float, default=18.0, help="Elapsed months since Course 1.")
    p_run.add_argument("--planned-fx", type=int, default=5, help="Planned Course-2 fraction count.")
    p_run.add_argument("--planned-rx", type=float, default=35.0, help="Planned Course-2 prescription (Gy).")
    p_run.add_argument("--tre-sigma", type=float, default=2.5, help="DIR uncertainty TRE sigma (mm).")
    p_run.add_argument("--ensemble", type=int, default=24, help="Number of diffeomorphic DIR ensemble members.")
    p_run.add_argument("--out-dir", default="outputs", help="Root output directory for figures and reports.")

    p_lookup = sub.add_parser("lookup", help="Look up TG-263 OAR normalization & clinical constraints.")
    p_lookup.add_argument("oar_name", help="Raw or TG-263 OAR name.")

    sub.add_parser("mcp", help="Start JSON-RPC 2.0 MCP stdio server for frontier agent harnesses.")

    args = parser.parse_args()

    if args.command == "mcp":
        serve_stdio()
        return

    if args.command == "lookup":
        norm = normalize_tg263(args.oar_name)
        cons = get_oar_constraint(norm)
        if cons is None:
            print(json.dumps({"normalized": norm, "found": False}, indent=2))
        else:
            print(json.dumps(cons.model_dump(mode="json"), indent=2))
        return

    if args.command == "run":
        out_root = Path(args.out_dir)
        fig_dir = out_root / "figures"
        rep_dir = out_root / "reports"
        fig_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)

        patient = load_openkbp_patient(args.patient_dir)
        case = build_reirradiation_benchmark_from_openkbp(
            patient,
            months_elapsed=args.months_elapsed,
            planned_n_fractions=args.planned_fx,
            planned_rx_gy=args.planned_rx,
        )
        report = run_frontier_investigation(
            case,
            tre_sigma_mm=args.tre_sigma,
            n_ensemble=args.ensemble,
        )

        # Write ESTRO/EORTC Markdown dossier + JSON provenance ledger
        dossier_md = emit_estro_eortc_markdown(report)
        md_path = rep_dir / "ESTRO_EORTC_ReIrradiation_Dossier.md"
        md_path.write_text(dossier_md, encoding="utf-8")

        ledger_payload = {
            "baseline_result": report.baseline.result.model_dump(mode="json"),
            "tipping_points": [tp.model_dump(mode="json") for tp in report.tipping_points],
            "schedule_matrix": [r.model_dump(mode="json") for r in report.schedule_matrix],
            "critic_audit": report.critic_audit.model_dump(mode="json"),
        }
        json_path = rep_dir / "provenance_audit_ledger.json"
        json_path.write_text(json.dumps(ledger_payload, indent=2), encoding="utf-8")

        figs = generate_all_publication_figures(case, report, fig_dir)

        print(f"Case: {case.case_id}")
        print(f"Global Verdict: {report.baseline.result.global_verdict.value}")
        print(f"Critic-Verifier Audit: {'PASS' if report.critic_audit.passed else 'FAIL'}")
        print(f"Max Forward-Inverse Residual: {report.baseline.result.provenance.max_inverse_residual_gy:.2e} Gy")
        print(f"Dossier: {md_path}")
        print(f"Ledger:  {json_path}")
        print(f"Figures ({len(figs)}): {', '.join(f.name for f in figs)}")


if __name__ == "__main__":
    main()
