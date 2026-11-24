"""Command-Line Interface for reirt-biobudget."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reirt_biobudget_mpl")

import argparse
import json
from pathlib import Path

from reirt_biobudget.budget_solver import solve_reirradiation_budget
from reirt_biobudget.cohort import run_openkbp_cohort_benchmark
from reirt_biobudget.dicom_io import (
    export_budget_iso_rtstruct,
    export_remaining_budget_dicom_rtdose,
    write_dicom_deformable_reg,
)
from reirt_biobudget.dir_envelope import warp_array
from reirt_biobudget.harness.agent_orchestrator import (
    emit_estro_eortc_markdown,
    run_frontier_investigation,
)
from reirt_biobudget.harness.mcp_server import serve_stdio
from reirt_biobudget.io_loaders import (
    build_reirradiation_benchmark_from_openkbp,
    load_openkbp_patient,
)
from reirt_biobudget.ir import Volume3D
from reirt_biobudget.legacy_reconstructor import (
    generate_legacy_summary_text_from_ground_truth,
    parse_legacy_treatment_summary,
    solve_legacy_missing_dicom_budget,
)
from reirt_biobudget.ntcp_sobol import evaluate_oar_ntcp_envelope, run_joint_sobol_sensitivity
from reirt_biobudget.ontology import get_oar_constraint, normalize_tg263
from reirt_biobudget.visualization import (
    generate_all_publication_figures,
    plot_fig6_cohort_landmark_validation,
    plot_fig7_sobol_ntcp_analysis,
    plot_fig8_missing_dicom_reconstruction,
)


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
    p_run.add_argument("--export-dicom", action="store_true", help="Export DICOM REG, RTDOSE & iso-RTSTRUCT.")
    p_run.add_argument("--out-dir", default="outputs", help="Root output directory for figures and reports.")

    p_coh = sub.add_parser(
        "cohort",
        help="Run multi-patient OpenKBP cohort + landmark TRE + K=2 + Sobol/NTCP + DICOM export.",
    )
    p_coh.add_argument("--data-root", default="data/openkbp", help="Directory containing OpenKBP pt_* folders.")
    p_coh.add_argument("--patients", nargs="+", default=["pt_1", "pt_2", "pt_3"], help="Patient IDs.")
    p_coh.add_argument("--tre-sigma", type=float, default=2.5, help="DIR uncertainty TRE sigma (mm).")
    p_coh.add_argument("--ensemble", type=int, default=16, help="Ensemble size per patient.")
    p_coh.add_argument("--sobol-pow2", type=int, default=8, help="Sobol base sample exponent (2^m).")
    p_coh.add_argument("--out-dir", default="outputs", help="Root output directory.")

    p_lookup = sub.add_parser("lookup", help="Look up TG-263 OAR normalization & clinical constraints.")
    p_lookup.add_argument("oar_name", help="Raw or TG-263 OAR name.")

    p_leg = sub.add_parser(
        "legacy-reconstruct",
        help="Reconstruct 3D prior-dose envelope from legacy PDF/text summary (no prior DICOM) & run blinded OpenKBP validation.",
    )
    p_leg.add_argument("--data-root", default="data/openkbp", help="Directory containing OpenKBP pt_* folders.")
    p_leg.add_argument("--patients", nargs="+", default=["pt_1", "pt_2", "pt_3"], help="Patient IDs for blinded validation.")
    p_leg.add_argument("--summary-file", default=None, help="Optional outside-hospital summary .txt file to parse for first patient.")
    p_leg.add_argument("--planned-fx", type=int, default=5, help="Planned Course-2 fraction count.")
    p_leg.add_argument("--planned-rx", type=float, default=35.0, help="Planned Course-2 prescription (Gy).")
    p_leg.add_argument("--out-dir", default="outputs", help="Root output directory.")

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

        if args.export_dicom:
            dcm_dir = out_root / "dicom"
            write_dicom_deformable_reg(
                case.nominal_dvf_mm,
                case.ct_course2.spacing_mm,
                case.ct_course2.origin_mm,
                case.planned_course.frame_of_reference_uid,
                case.prior_course.frame_of_reference_uid,
                dcm_dir / f"{case.patient_id}_svf_reg.dcm",
                patient_id=case.patient_id,
            )
            export_remaining_budget_dicom_rtdose(
                report.baseline.remaining_physical_p95_map,
                dcm_dir / f"{case.patient_id}_remaining_budget_p95_rtdose.dcm",
                patient_id=case.patient_id,
            )
            export_budget_iso_rtstruct(
                report.baseline.remaining_physical_p95_map,
                case.body_mask_course2,
                dcm_dir / f"{case.patient_id}_remaining_budget_shells_rtstruct.dcm",
                patient_id=case.patient_id,
            )

        print(f"Case: {case.case_id}")
        print(f"Global Verdict: {report.baseline.result.global_verdict.value}")
        print(f"Critic-Verifier Audit: {'PASS' if report.critic_audit.passed else 'FAIL'}")
        print(f"Max Forward-Inverse Residual: {report.baseline.result.provenance.max_inverse_residual_gy:.2e} Gy")
        print(f"Dossier: {md_path}")
        print(f"Ledger:  {json_path}")
        print(f"Figures ({len(figs)}): {', '.join(f.name for f in figs)}")
        return

    if args.command == "cohort":
        out_root = Path(args.out_dir)
        fig_dir = out_root / "figures"
        rep_dir = out_root / "reports"
        fig_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)

        data_root = Path(args.data_root)
        pdirs = [data_root / pid for pid in args.patients]
        cohort_res = run_openkbp_cohort_benchmark(
            pdirs,
            tre_sigma_mm=args.tre_sigma,
            n_ensemble=args.ensemble,
            sobol_pow2=args.sobol_pow2,
        )

        # Run detailed Sobol across 3 serial OARs + LKB NTCP on first patient for Fig 7
        p0 = load_openkbp_patient(pdirs[0])
        c0 = build_reirradiation_benchmark_from_openkbp(p0)
        out0 = solve_reirradiation_budget(
            case_id=c0.case_id,
            prior_course=c0.prior_course,
            prior_dose=c0.prior_dose_course1,
            planned_course=c0.planned_course,
            nominal_dvf_mm=c0.nominal_dvf_mm,
            nominal_inv_dvf_mm=c0.nominal_inv_dvf_mm,
            oar_masks_course2=c0.oar_masks_course2,
            body_mask_course2=c0.body_mask_course2,
            candidate_course2_dose=c0.candidate_dose_course2,
            tre_sigma_mm=args.tre_sigma,
            n_ensemble=max(args.ensemble, 20),
            retain_ensemble_samples=True,
        )
        sobol_list = []
        for oar_name in ["SpinalCord", "Brainstem", "Bone_Mandible"]:
            if oar_name in c0.oar_masks_course2:
                m = c0.oar_masks_course2[oar_name]
                sobol_list.append(
                    run_joint_sobol_sensitivity(
                        oar_name=oar_name,
                        prior_ensemble_vox=out0.dir_envelope.extract_masked_ensemble(m),
                        candidate_course2_vox=c0.candidate_dose_course2.data[m],
                        prior_course=c0.prior_course,
                        planned_course=c0.planned_course,
                        voxel_vol_cc=c0.prior_dose_course1.voxel_volume_cc,
                        output_mode="eqd2",
                        n_base_pow2=args.sobol_pow2,
                    )
                )
        ntcp_list = [
            ns
            for oar_name, env_d in out0.oar_eqd2_envelopes.items()
            if (ns := evaluate_oar_ntcp_envelope(oar_name, env_d)) is not None
        ]

        f6 = plot_fig6_cohort_landmark_validation(cohort_res, fig_dir)
        f7 = plot_fig7_sobol_ntcp_analysis(sobol_list, ntcp_list, fig_dir)

        cohort_json = {
            "patients": [p.model_dump(mode="json") for p in cohort_res.patients],
            "k2_multicourse": (
                cohort_res.k2_multicourse_output.result.model_dump(mode="json")
                if cohort_res.k2_multicourse_output is not None
                else None
            ),
            "pt1_sobol_serial_oars": [s.model_dump(mode="json") for s in sobol_list],
            "pt1_lkb_ntcp": [n.model_dump(mode="json") for n in ntcp_list],
        }
        rep_path = rep_dir / "cohort_sobol_ntcp_benchmark.json"
        rep_path.write_text(json.dumps(cohort_json, indent=2), encoding="utf-8")

        print(f"Cohort Patients Validated: {len(cohort_res.patients)}")
        for p in cohort_res.patients:
            lv = p.landmark_validation
            print(
                f"  - {p.patient_id}: Pre-DIR TRE={lv.pre_dir_tre_mean_mm:.2f} mm -> "
                f"Post-DIR TRE={lv.post_dir_tre_p95_mm:.4f} mm | "
                f"90% Coverage={lv.empirical_90pct_envelope_coverage_pct:.1f}% | "
                f"TG-132={'PASS' if p.tg132_pass else 'WARN'}"
            )
        if cohort_res.k2_multicourse_output is not None:
            k2 = cohort_res.k2_multicourse_output.result
            print(
                f"  - K=2 Multi-Course ({k2.case_id}): {len(k2.prior_courses)} prior courses | "
                f"Residual={k2.provenance.max_inverse_residual_gy:.2e} Gy | Verdict={k2.global_verdict.value}"
            )
        print(f"Report:  {rep_path}")
        print(f"Figures: {f6.name}, {f7.name}")
        return

    if args.command == "legacy-reconstruct":
        out_root = Path(args.out_dir)
        fig_dir = out_root / "figures"
        rep_dir = out_root / "reports"
        fig_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)

        data_root = Path(args.data_root)
        cohort_audits = []
        primary_case = None
        primary_out = None
        primary_truth = None
        primary_summary_text = ""

        for idx, pid in enumerate(args.patients):
            patient = load_openkbp_patient(data_root / pid)
            case = build_reirradiation_benchmark_from_openkbp(
                patient,
                planned_n_fractions=args.planned_fx,
                planned_rx_gy=args.planned_rx,
            )
            warped_truth = warp_array(
                case.prior_dose_course1.data,
                case.nominal_dvf_mm,
                case.ct_course2.spacing_mm,
                order=1,
                cval=0.0,
                clamp_nonnegative=True,
            )
            truth_vol = Volume3D(
                data=warped_truth,
                spacing_mm=case.ct_course2.spacing_mm,
                origin_mm=case.ct_course2.origin_mm,
                frame_of_reference_uid=case.planned_course.frame_of_reference_uid,
                units="Gy",
                name=f"{pid}_hidden_truth",
            )
            if idx == 0 and args.summary_file:
                raw_text = Path(args.summary_file).read_text(encoding="utf-8")
            else:
                raw_text = generate_legacy_summary_text_from_ground_truth(
                    patient_id=pid,
                    prior_dose=case.prior_dose_course1,
                    oar_masks=case.oar_masks_course1,
                )
            rec = parse_legacy_treatment_summary(raw_text, record_id=f"LEGACY_{pid.upper()}")
            approx_ptv = (
                warp_array(
                    patient.ptv_masks["PTV70"].astype(float),
                    case.nominal_dvf_mm,
                    case.ct_course2.spacing_mm,
                    order=1,
                    cval=0.0,
                )
                > 0.5
                if "PTV70" in patient.ptv_masks
                else None
            )
            out = solve_legacy_missing_dicom_budget(
                case_id=f"{case.case_id}-NO-DICOM",
                legacy_record=rec,
                planned_course=case.planned_course,
                body_mask_current=case.body_mask_course2,
                oar_masks_current=case.oar_masks_course2,
                spacing_mm=case.ct_course2.spacing_mm,
                origin_mm=case.ct_course2.origin_mm,
                approx_prior_ptv_current=approx_ptv,
                hidden_ground_truth_dose=truth_vol,
            )
            cohort_audits.append((pid, out.audit))
            if idx == 0:
                primary_case = case
                primary_out = out
                primary_truth = truth_vol
                primary_summary_text = raw_text

        assert primary_case is not None and primary_out is not None and primary_truth is not None
        f8 = plot_fig8_missing_dicom_reconstruction(
            primary_case, primary_out, primary_truth, cohort_audits, fig_dir
        )

        payload = {
            "primary_legacy_summary_text": primary_summary_text,
            "cohort_blinded_audits": {
                pid: aud.model_dump(mode="json") for pid, aud in cohort_audits
            },
        }
        rep_path = rep_dir / "legacy_missing_dicom_audit.json"
        rep_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(f"Missing-DICOM Blinded Validation Completed ({len(cohort_audits)} patients):")
        for pid, aud in cohort_audits:
            sc = next((s for s in aud.oar_summaries if s.tg263_name == "SpinalCord"), None)
            sc_str = (
                f"SpinalCord Legacy={sc.legacy_remaining_physical_ceiling_gy:.1f} Gy vs "
                f"DICOM={sc.true_dicom_remaining_physical_ceiling_gy:.1f} Gy "
                f"(Value={sc.dicom_recovery_value_gy:+.1f} Gy, Safe={sc.envelope_safely_bounds_truth})"
                if sc
                else ""
            )
            print(
                f"  - {pid}: FitErr={aud.max_constraint_fit_error_gy:.3f} Gy | "
                f"UpperSafety={aud.truth_voxel_upper_safety_rate_pct:.1f}% | "
                f"InvRes={aud.max_inverse_residual_gy:.2e} Gy | {sc_str}"
            )
        print(f"Report: {rep_path}")
        print(f"Figure: {f8.name}")


if __name__ == "__main__":
    main()
