"""Publication-Grade Academic Visualizations for reirt-biobudget.

Generates 300-DPI journal-ready figures (PNG + vector PDF) adhering to Medical Physics /
Radiotherapy and Oncology / Nature visual standards:
- Radiological CT orientation with physical millimeter aspect ratios & scalebars
- Colorblind-safe Okabe-Ito & perceptually uniform colormaps
- Bold lowercase panel labels (a, b, c, ...) and full data provenance footers
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reirt_biobudget_mpl")
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from reirt_biobudget.budget_solver import compute_dvh_curve
from reirt_biobudget.harness.agent_orchestrator import AdversarialSweepReport
from reirt_biobudget.io_loaders import ReIrradiationBenchmarkCase

# Okabe-Ito / NPG colorblind-safe publication palette
OAR_COLORS: dict[str, str] = {
    "SpinalCord": "#D55E00",     # Vermilion
    "Brainstem": "#0072B2",      # Strong Blue
    "Bone_Mandible": "#009E73",  # Bluish Green
    "Parotid_L": "#CC79A7",      # Reddish Purple
    "Parotid_R": "#E69F00",      # Orange
    "Larynx": "#56B4E9",         # Sky Blue
    "Esophagus": "#332288",      # Indigo
}

PROVENANCE_FOOTER = (
    "Source data: OpenKBP public clinical dataset (pt_1; Babier et al., Med Phys 2021, DOI: 10.1002/mp.14845, CC BY 4.0) "
    "with documented SVF longitudinal benchmark. Research tool — not a medical device."
)


def _configure_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def _add_panel_label(ax: plt.Axes, label: str, dark_bg: bool = False) -> None:
    color = "white" if dark_bg else "#111111"
    ax.text(
        0.025,
        0.965,
        label,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        va="top",
        ha="left",
        color=color,
        bbox=(
            {"boxstyle": "round,pad=0.18", "facecolor": (0, 0, 0, 0.55), "edgecolor": "none"}
            if dark_bg
            else None
        ),
    )


def _find_best_slice_indices(case: ReIrradiationBenchmarkCase) -> tuple[int, int]:
    """Select axial (z) and sagittal (x) slice indices intersecting recurrent PTV and SpinalCord."""
    if "SpinalCord" in case.oar_masks_course2:
        cord_z = np.where(np.any(case.oar_masks_course2["SpinalCord"], axis=(1, 2)))[0]
        ptv_z = np.where(np.any(case.recurrent_ptv_course2, axis=(1, 2)))[0]
        overlap_z = np.intersect1d(cord_z, ptv_z)
        if overlap_z.size > 0:
            z_idx = int(np.median(overlap_z))
        else:
            z_idx = int(np.median(ptv_z)) if ptv_z.size > 0 else case.ct_course2.shape[0] // 2
        # Sagittal slice near midline / SpinalCord x-centroid
        cord_x = np.where(case.oar_masks_course2["SpinalCord"])[2]
        x_idx = int(np.round(np.median(cord_x)))
    else:
        z_idx = case.ct_course2.shape[0] // 2
        x_idx = case.ct_course2.shape[2] // 2
    return z_idx, x_idx


def _body_crop_bounds(mask_2d: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    """Compute tight anatomical bounding box around body mask for crisp clinical framing."""
    rows, cols = np.where(mask_2d)
    if rows.size == 0:
        return 0, mask_2d.shape[0], 0, mask_2d.shape[1]
    r0 = max(int(rows.min()) - pad, 0)
    r1 = min(int(rows.max()) + pad + 1, mask_2d.shape[0])
    c0 = max(int(cols.min()) - pad, 0)
    c1 = min(int(cols.max()) + pad + 1, mask_2d.shape[1])
    return r0, r1, c0, c1


def _add_scalebar(ax: plt.Axes, length_mm: float, pixel_mm: float, color: str = "white") -> None:
    """Draw a crisp physical millimeter scalebar with high-contrast backing."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bar_px = length_mm / pixel_mm
    x_end = xlim[1] - 0.06 * (xlim[1] - xlim[0])
    x_start = x_end - bar_px
    y_pos = ylim[0] + 0.075 * (ylim[1] - ylim[0])  # ylim is inverted for imshow
    ax.plot([x_start, x_end], [y_pos, y_pos], color="black", linewidth=4.0, solid_capstyle="butt")
    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=2.0, solid_capstyle="butt")
    ax.text(
        0.5 * (x_start + x_end),
        y_pos - 0.035 * (ylim[1] - ylim[0]),
        f"{length_mm:.0f} mm",
        color=color,
        fontsize=7.2,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": (0, 0, 0, 0.62), "edgecolor": "none"},
    )


def plot_fig1_triplanar_biobudget(
    case: ReIrradiationBenchmarkCase,
    report: AdversarialSweepReport,
    out_dir: Path,
) -> Path:
    """Figure 1: Axial & Sagittal CT, Prior EQD2, DIR Uncertainty Spread, and 3D Remaining Budget."""
    _configure_academic_style()
    out = report.baseline
    dz, dy, dx = case.ct_course2.spacing_mm
    z_idx, x_idx = _find_best_slice_indices(case)

    ct_ax = case.ct_course2.data[z_idx]
    body_ax = case.body_mask_course2[z_idx]
    r0a, r1a, c0a, c1a = _body_crop_bounds(body_ax, pad=6)

    ct_sag = case.ct_course2.data[:, :, x_idx]
    body_sag = case.body_mask_course2[:, :, x_idx]
    r0s, r1s, c0s, c1s = _body_crop_bounds(body_sag, pad=6)

    prior_eqd2 = out.prior_eqd2_nominal_map.data
    spread_eqd2 = out.prior_eqd2_spread_map.data
    rem_phys = out.remaining_physical_p95_map.data * case.body_mask_course2

    fig = plt.figure(figsize=(14.6, 7.7))
    gs = GridSpec(2, 4, figure=fig, wspace=0.20, hspace=0.26, top=0.88, bottom=0.11)

    col_titles = [
        "Follow-Up CT & TG-263 Anatomy",
        "Prior Effective EQD2 (Gy)",
        "DIR Spread (p95 − p05, Gy)",
        "Remaining Budget p95 (Gy)",
    ]

    overlays = [
        None,
        (prior_eqd2, "inferno", 5.0, max(float(np.percentile(prior_eqd2[case.body_mask_course2], 99.5)), 60.0), "Prior EQD2 (Gy)"),
        (spread_eqd2, "plasma", 0.5, max(float(np.percentile(spread_eqd2[case.body_mask_course2], 99.5)), 8.0), "90% DIR Spread (Gy)"),
        (rem_phys, "viridis", 0.0, max(float(np.percentile(rem_phys[case.body_mask_course2], 98)), 25.0), "Remaining Physical Ceiling (Gy)"),
    ]

    row_configs = [
        ("Axial", z_idx, (r0a, r1a, c0a, c1a), dy / dx, dx, ["a", "b", "c", "d"]),
        ("Sagittal", x_idx, (r0s, r1s, c0s, c1s), dz / dy, dy, ["e", "f", "g", "h"]),
    ]

    for row_i, (plane, sl_idx, (r0, r1, c0, c1), aspect, px_mm, labels) in enumerate(row_configs):
        ct_sl = (ct_ax if plane == "Axial" else ct_sag)[r0:r1, c0:c1]

        for col_i in range(4):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.imshow(
                ct_sl,
                cmap="gray",
                vmin=-250.0,
                vmax=450.0,
                aspect=aspect,
                interpolation="bilinear",
            )

            ov = overlays[col_i]
            im_ov = None
            if ov is not None:
                vol, cmap_name, vmin, vmax, _cbar_label = ov
                sl_data = (vol[sl_idx] if plane == "Axial" else vol[:, :, sl_idx])[r0:r1, c0:c1]
                body_sl = (body_ax if plane == "Axial" else body_sag)[r0:r1, c0:c1]
                if col_i == 3:
                    masked = np.ma.masked_where(~body_sl, sl_data)
                else:
                    masked = np.ma.masked_where((~body_sl) | (sl_data < vmin), sl_data)
                im_ov = ax.imshow(
                    masked,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.72 if col_i < 3 else 0.78,
                    aspect=aspect,
                    interpolation="bilinear",
                )

            # Overlay OAR contours & recurrent PTV
            for oar_name, mask3d in case.oar_masks_course2.items():
                msl = (mask3d[sl_idx] if plane == "Axial" else mask3d[:, :, sl_idx])[r0:r1, c0:c1]
                if np.any(msl):
                    ax.contour(
                        msl.astype(float),
                        levels=[0.5],
                        colors=[OAR_COLORS.get(oar_name, "#FFFFFF")],
                        linewidths=1.35,
                    )
            ptv_sl = (
                case.recurrent_ptv_course2[sl_idx]
                if plane == "Axial"
                else case.recurrent_ptv_course2[:, :, sl_idx]
            )[r0:r1, c0:c1]
            if np.any(ptv_sl):
                ax.contour(
                    ptv_sl.astype(float),
                    levels=[0.5],
                    colors=["#FF2A55"],
                    linewidths=1.5,
                    linestyles="--",
                )

            if row_i == 0:
                ax.set_title(col_titles[col_i], pad=7)
            if col_i == 0:
                orient = "R → L (Axial)" if plane == "Axial" else "A → P (Sagittal)"
                ax.set_ylabel(f"{plane} Slice #{sl_idx}\n({orient})", fontweight="bold", fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
            _add_panel_label(ax, labels[col_i], dark_bg=True)
            if col_i == 0:
                _add_scalebar(ax, 30.0, px_mm, color="white")

            if row_i == 1 and im_ov is not None:
                cbar = fig.colorbar(im_ov, ax=ax, orientation="horizontal", fraction=0.055, pad=0.06)
                cbar.set_label(ov[4], fontsize=8.0)
                cbar.ax.tick_params(labelsize=7.5)

    # Shared contour legend on top
    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="#FF2A55", linestyle="--", linewidth=1.6, label="Recurrent Salvage PTV"),
    ]
    for oar_name in ["SpinalCord", "Brainstem", "Bone_Mandible", "Parotid_L", "Parotid_R"]:
        if oar_name in case.oar_masks_course2:
            legend_handles.append(
                mpatches.Patch(
                    facecolor="none",
                    edgecolor=OAR_COLORS[oar_name],
                    linewidth=1.6,
                    label=oar_name,
                )
            )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(legend_handles),
        frameon=True,
        fontsize=8.2,
    )
    fig.suptitle(
        f"3D Biophysical Dose Accumulation, DIR-Uncertainty Envelope & Remaining Physical Budget ({case.case_id})",
        fontsize=12.0,
        fontweight="bold",
        y=0.995,
    )
    fig.text(0.5, 0.015, PROVENANCE_FOOTER, ha="center", fontsize=7.2, color="#555555", style="italic")

    png_path = out_dir / "fig1_triplanar_biobudget.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig1_triplanar_biobudget.pdf")
    plt.close(fig)
    return png_path


def plot_fig2_uncertainty_band_dvh(
    case: ReIrradiationBenchmarkCase,
    report: AdversarialSweepReport,
    out_dir: Path,
) -> Path:
    """Figure 2: Uncertainty-Band Cumulative EQD2 DVHs + OAR Dose Budget Decomposition."""
    _configure_academic_style()
    out = report.baseline
    summaries = out.result.oar_summaries
    dose_grid = np.linspace(0.0, 95.0, 380)

    fig = plt.figure(figsize=(13.6, 5.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.22, 1.0], wspace=0.25, top=0.86, bottom=0.16)

    # Panel (a): Cumulative EQD2 DVHs with 90% DIR uncertainty ribbons
    ax1 = fig.add_subplot(gs[0, 0])
    for s in summaries:
        oar = s.tg263_name
        env = out.oar_eqd2_envelopes.get(oar)
        if env is None:
            continue
        color = OAR_COLORS.get(oar, "#333333")
        dvh_p05 = compute_dvh_curve(env["cum_p05"], dose_grid)
        dvh_nom = compute_dvh_curve(env["cum_nominal"], dose_grid)
        dvh_p95 = compute_dvh_curve(env["cum_p95"], dose_grid)
        dvh_worst = compute_dvh_curve(env["cum_worst"], dose_grid)

        ax1.fill_between(dose_grid, dvh_p05, dvh_p95, color=color, alpha=0.20, linewidth=0)
        ax1.plot(dose_grid, dvh_nom, color=color, linewidth=2.0, label=f"{oar} ({s.metric_name})")
        ax1.plot(dose_grid, dvh_worst, color=color, linewidth=0.9, linestyle=":", alpha=0.85)

        # Draw subtle tick marker for OAR tolerance limit along top
        if oar in ("SpinalCord", "Brainstem"):
            ax1.axvline(s.eqd2_limit_gy, color=color, linestyle="--", linewidth=1.0, alpha=0.65)
            ax1.text(
                s.eqd2_limit_gy + 0.8,
                96.0 if oar == "SpinalCord" else 84.0,
                f"{oar}\nLimit ({s.eqd2_limit_gy:.0f} Gy)",
                color=color,
                fontsize=7.2,
                fontweight="bold",
                va="top",
            )

    ax1.set_xlim(0, 95)
    ax1.set_ylim(0, 102)
    ax1.set_xlabel("Cumulative Effective EQD2 (Gy)", fontweight="bold")
    ax1.set_ylabel("Relative OAR Volume (%)", fontweight="bold")
    ax1.set_title("Cumulative EQD2 DVHs with 90% DIR Uncertainty Envelopes", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Extra legend entries for band & worst-case style
    handles, _labels = ax1.get_legend_handles_labels()
    handles.append(mpatches.Patch(facecolor="#888888", alpha=0.25, label="90% DIR band [p05, p95]"))
    handles.append(plt.Line2D([0], [0], color="#444444", linestyle=":", linewidth=1.1, label="Ensemble worst-case"))
    ax1.legend(handles=handles, loc="lower left", ncol=2, fontsize=7.4)
    _add_panel_label(ax1, "a")

    # Panel (b): OAR Prior vs. Cumulative EQD2 Dot-Whisker against Caution & Limit
    ax2 = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(summaries))
    for i, s in enumerate(summaries):
        # Background tolerance bar (safe -> caution -> limit)
        ax2.barh(i, s.eqd2_caution_gy, height=0.52, color="#E8F5E9", edgecolor="none", zorder=1)
        ax2.barh(
            i,
            s.eqd2_limit_gy - s.eqd2_caution_gy,
            left=s.eqd2_caution_gy,
            height=0.52,
            color="#FFF8E1",
            edgecolor="none",
            zorder=1,
        )
        ax2.plot([s.eqd2_limit_gy, s.eqd2_limit_gy], [i - 0.30, i + 0.30], color="#C62828", linewidth=2.0, zorder=4)

        # Prior Effective [p05, p95] whisker
        ax2.plot(
            [s.prior_eqd2_effective_p05_gy, s.prior_eqd2_effective_p95_gy],
            [i + 0.10, i + 0.10],
            color="#546E7A",
            linewidth=2.4,
            zorder=5,
        )
        ax2.scatter(
            [s.prior_eqd2_effective_nominal_gy],
            [i + 0.10],
            color="#37474F",
            s=36,
            zorder=6,
            label="Prior Eff. EQD2 (Nom, [p05–p95])" if i == 0 else None,
        )

        # Cumulative Candidate Course 2 (Nom -> p95)
        if s.cumulative_eqd2_nominal_gy is not None and s.cumulative_eqd2_p95_gy is not None:
            v_col = (
                "#C62828"
                if s.verdict.value == "VIOLATION"
                else ("#EF6C00" if s.verdict.value == "CAUTION" else "#2E7D32")
            )
            ax2.plot(
                [s.cumulative_eqd2_nominal_gy, s.cumulative_eqd2_p95_gy],
                [i - 0.12, i - 0.12],
                color=v_col,
                linewidth=2.8,
                zorder=5,
            )
            ax2.scatter(
                [s.cumulative_eqd2_p95_gy],
                [i - 0.12],
                color=v_col,
                marker="D",
                s=42,
                zorder=7,
                label="Cumulative p95 EQD2 (with Course 2)" if i == 0 else None,
            )
            ax2.text(
                max(s.cumulative_eqd2_p95_gy, s.eqd2_limit_gy) + 1.8,
                i,
                f"{s.verdict.value}\nRem: {s.remaining_physical_p95_gy:.1f} Gy",
                va="center",
                fontsize=7.3,
                fontweight="bold",
                color=v_col,
            )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{s.tg263_name}\n({s.metric_name})" for s in summaries], fontsize=8.2)
    ax2.invert_yaxis()
    max_x = max(max(s.eqd2_limit_gy, s.cumulative_eqd2_p95_gy or 0.0) for s in summaries) + 26.0
    ax2.set_xlim(0, max_x)
    ax2.set_xlabel("Equivalent Dose in 2 Gy Fractions — EQD2 (Gy)", fontweight="bold")
    ax2.set_title("OAR Tolerance Headroom & Remaining Physical Budget", pad=26)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    p_handles = [
        mpatches.Patch(facecolor="#E8F5E9", label="Safe"),
        mpatches.Patch(facecolor="#FFF8E1", label="Caution"),
        plt.Line2D([0], [0], color="#C62828", linewidth=2.0, label="Limit"),
        plt.Line2D([0], [0], color="#37474F", marker="o", linewidth=2.0, markersize=4.5, label="Prior [p05–p95]"),
        plt.Line2D([0], [0], color="#EF6C00", marker="D", linewidth=2.0, markersize=4.5, label="Cum. p95"),
    ]
    ax2.legend(
        handles=p_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5,
        fontsize=6.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    _add_panel_label(ax2, "b")

    fig.suptitle(
        f"Biophysical Cumulative EQD2 DVHs & OAR Safety Headroom ({out.result.planned_course.label})",
        fontsize=11.8,
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.5, 0.025, PROVENANCE_FOOTER, ha="center", fontsize=7.2, color="#555555", style="italic")

    png_path = out_dir / "fig2_uncertainty_band_dvh.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig2_uncertainty_band_dvh.pdf")
    plt.close(fig)
    return png_path


def plot_fig3_adversarial_tipping_phase(
    case: ReIrradiationBenchmarkCase,
    report: AdversarialSweepReport,
    out_dir: Path,
) -> Path:
    """Figure 3: Adversarial Assumption Sensitivity Tornado + 2D Safety Phase Diagram."""
    _configure_academic_style()
    tps = report.tipping_points
    pd = report.phase_diagram

    fig = plt.figure(figsize=(13.8, 5.7))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.26, top=0.79, bottom=0.15)

    # Panel (a): Stacked Horizontal Sensitivity Decomposition per OAR
    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(tps))
    sw_rec = np.array([t.swing_recovery_gy for t in tps])
    sw_tre = np.array([t.swing_dir_tre_gy for t in tps])
    sw_ab = np.array([t.swing_alpha_beta_gy for t in tps])
    sw_mod = np.array([t.swing_model_regime_gy for t in tps])

    h = 0.54
    ax1.barh(y, sw_rec, height=h, color="#0072B2", label="Tissue Recovery Credit (0% to max)")
    ax1.barh(y, sw_tre, left=sw_rec, height=h, color="#D55E00", label="DIR TRE Uncertainty (0 to 4 mm)")
    ax1.barh(y, sw_ab, left=sw_rec + sw_tre, height=h, color="#009E73", label=r"Radiobiology $\alpha/\beta$ Range")
    ax1.barh(y, sw_mod, left=sw_rec + sw_tre + sw_ab, height=h, color="#CC79A7", label="Model Regime (LQ vs. LQL)")

    totals = sw_rec + sw_tre + sw_ab + sw_mod
    short_driver_map = {
        "Tissue Recovery Credit": "Recovery",
        "DIR Registration TRE (0-4 mm)": "DIR TRE",
        "Radiobiology alpha/beta Uncertainty": "alpha/beta",
        "High-Dose Model Regime (LQ vs LQL)": "LQ vs LQL",
    }
    for i, t in enumerate(tps):
        flag = " [FLIPS]" if t.flips_to_violation else ""
        top_val = max(
            t.swing_recovery_gy,
            t.swing_dir_tre_gy,
            t.swing_alpha_beta_gy,
            t.swing_model_regime_gy,
        )
        short_d = short_driver_map.get(t.dominant_driver, t.dominant_driver)
        ax1.text(
            totals[i] + 0.35,
            i,
            f"Top: {short_d} ({top_val:.1f} Gy){flag}",
            va="center",
            fontsize=7.4,
            fontweight="bold" if t.flips_to_violation else "normal",
            color="#B71C1C" if t.flips_to_violation else "#222222",
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{t.oar_name}\n({t.metric_name})" for t in tps], fontsize=8.2)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(float(np.max(totals)) * 1.62, 18.0))
    ax1.set_xlabel("Cumulative EQD2 Sensitivity Swing — Sum of Factor Ranges (Gy)", fontweight="bold")
    ax1.set_title("Adversarial Assumption Sensitivity Decomposition", pad=42)
    ax1.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        fontsize=7.2,
        columnspacing=1.0,
    )
    _add_panel_label(ax1, "a")

    # Panel (b): 2D Safety Phase Diagram (Recovery % x DIR TRE mm)
    ax2 = fig.add_subplot(gs[0, 1])
    R, T = np.meshgrid(pd.recovery_grid_pct, pd.tre_grid_mm)
    Z = pd.cum_eqd2_p95_matrix

    norm = mcolors.TwoSlopeNorm(
        vmin=min(float(np.min(Z)), pd.eqd2_caution_gy - 10.0),
        vcenter=pd.eqd2_caution_gy,
        vmax=max(float(np.max(Z)), pd.eqd2_limit_gy + 8.0),
    )
    cf = ax2.contourf(R, T, Z, levels=20, cmap="RdYlGn_r", norm=norm, alpha=0.90)
    cbar = fig.colorbar(cf, ax=ax2, pad=0.03)
    cbar.set_label(f"{pd.oar_name} Cumulative p95 EQD2 (Gy)", fontsize=8.3)

    # Exact clinical phase boundaries
    if float(np.min(Z)) < pd.eqd2_caution_gy < float(np.max(Z)):
        cs_c = ax2.contour(R, T, Z, levels=[pd.eqd2_caution_gy], colors=["#E65100"], linewidths=2.0, linestyles="--")
        ax2.clabel(cs_c, fmt=f"Caution ({pd.eqd2_caution_gy:.0f} Gy)", fontsize=7.8)
    if float(np.min(Z)) < pd.eqd2_limit_gy < float(np.max(Z)):
        cs_l = ax2.contour(R, T, Z, levels=[pd.eqd2_limit_gy], colors=["#8B0000"], linewidths=2.4)
        ax2.clabel(cs_l, fmt=f"Limit ({pd.eqd2_limit_gy:.0f} Gy)", fontsize=8.0)

    # Mark baseline operating point
    base_s = next(s for s in report.baseline.result.oar_summaries if s.tg263_name == pd.oar_name)
    base_r_pct = base_s.recovery_fraction_applied * 100.0
    base_tre = report.baseline.result.dir_quality.ensemble_tre_sigma_mm
    ax2.scatter(
        [base_r_pct],
        [base_tre],
        color="white",
        edgecolor="#111111",
        s=95,
        marker="*",
        linewidth=1.2,
        zorder=8,
        label=f"Baseline ({base_r_pct:.1f}% rec., {base_tre:.1f} mm TRE)",
    )
    ax2.set_xlabel("Assumed Prior OAR Recovery Credit (%)", fontweight="bold")
    ax2.set_ylabel(r"DIR Uncertainty $\sigma_{\mathrm{TRE}}$ (mm)", fontweight="bold")
    ax2.set_title(f"{pd.oar_name} Clinical Safety Phase Diagram", pad=8)
    ax2.legend(loc="upper right", fontsize=7.6)
    _add_panel_label(ax2, "b")

    fig.suptitle(
        "Frontier Agent Harness: Adversarial Failure-Mode Discovery & Clinical Decision Boundaries",
        fontsize=11.8,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.5, 0.025, PROVENANCE_FOOTER, ha="center", fontsize=7.2, color="#555555", style="italic")

    png_path = out_dir / "fig3_adversarial_tipping_phase.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig3_adversarial_tipping_phase.pdf")
    plt.close(fig)
    return png_path


def plot_fig4_schedule_modality_matrix(
    report: AdversarialSweepReport,
    out_dir: Path,
) -> Path:
    """Figure 4: Multi-Schedule & Modality Remaining Physical Dose Ceiling & Limiting Margin."""
    _configure_academic_style()
    rows = report.schedule_matrix
    compact_names = [
        "35/5 SBRT\n(Photon LQ)",
        "35/5 SBRT\n(Photon LQL)",
        "45/15 Hypo\n(IMRT LQ)",
        "50/25 Conv\n(IMRT LQ)",
        "50/25 Proton\n(McNamara)",
        "60/50 BID\n(Lea–Catch.)",
    ]
    short_labels = compact_names[: len(rows)]
    key_oars = [o for o in ["SpinalCord", "Brainstem", "Bone_Mandible"] if o in rows[0].oar_remaining_physical_p95_gy]

    fig = plt.figure(figsize=(14.4, 5.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.22, 1.05], wspace=0.40, top=0.84, bottom=0.20)

    # Panel (a): Grouped Bar Chart of Remaining Physical Dose Ceiling across 6 schedules
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(rows))
    width = 0.25
    max_bar = 0.0
    for oi, oar in enumerate(key_oars):
        vals = [r.oar_remaining_physical_p95_gy[oar] for r in rows]
        max_bar = max(max_bar, max(vals))
        offset = (oi - (len(key_oars) - 1) / 2.0) * width
        bars = ax1.bar(
            x + offset,
            vals,
            width=width * 0.92,
            color=OAR_COLORS.get(oar, "#555555"),
            label=oar,
            edgecolor="white",
            linewidth=0.6,
        )
        for b, v in zip(bars, vals):
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                v + 0.7,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=7.5, ha="center")
    ax1.set_ylim(0, max_bar * 1.18)
    ax1.set_ylabel("Safe Remaining Physical Dose Ceiling p95 (Gy)", fontweight="bold")
    ax1.set_title("Exact Inverted Course-2 Physical Dose Ceiling by Schedule", pad=10)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.07, 0.99), fontsize=7.8, ncol=3)
    _add_panel_label(ax1, "a")

    # Panel (b): Limiting Serial OAR EQD2 Margin (Positive = Safe Headroom, Negative = Violation)
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(rows))
    margins = np.array([r.limiting_oar_margin_eqd2_gy for r in rows])
    bar_colors = [
        "#C62828" if r.verdict.value == "VIOLATION" else ("#EF6C00" if r.verdict.value == "CAUTION" else "#2E7D32")
        for r in rows
    ]
    ax2.barh(y, margins, height=0.52, color=bar_colors, alpha=0.88, edgecolor="#222222", linewidth=0.5)
    ax2.axvline(0.0, color="#111111", linewidth=1.2)

    x_lo = min(float(np.min(margins)) - 6.0, -8.0)
    x_hi = max(float(np.max(margins)) + 26.0, 26.0)
    for i, r in enumerate(rows):
        m_val = r.limiting_oar_margin_eqd2_gy
        txt_x = max(m_val, 0.0) + 0.8
        ax2.text(
            txt_x,
            i,
            f"{m_val:+.1f} Gy ({r.limiting_oar}, {r.verdict.value})",
            va="center",
            ha="left",
            fontsize=7.4,
            fontweight="bold",
            color=bar_colors[i],
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels([s.replace("\n", " ") for s in short_labels], fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_xlabel("Serial OAR Margin: Limit − Cum. p95 EQD2 (Gy)", fontweight="bold")
    ax2.set_title("Critical Serial-OAR Safety Margin & Schedule Triage", pad=10)
    ax2.grid(True, axis="x", linestyle="--", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    _add_panel_label(ax2, "b")

    fig.suptitle(
        "Multi-Schedule & Multi-Modality Co-Design (Photon LQ / LQL / BID Lea–Catcheside vs. Variable-RBE Proton)",
        fontsize=11.4,
        fontweight="bold",
        y=0.96,
    )
    fig.text(0.5, 0.025, PROVENANCE_FOOTER, ha="center", fontsize=7.2, color="#555555", style="italic")

    png_path = out_dir / "fig4_schedule_modality_matrix.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig4_schedule_modality_matrix.pdf")
    plt.close(fig)
    return png_path


def plot_fig5_tg132_dir_forensics(
    case: ReIrradiationBenchmarkCase,
    report: AdversarialSweepReport,
    out_dir: Path,
) -> Path:
    """Figure 5: AAPM TG-132 DIR Quality Forensics (Jacobian Spatial Map, Histogram, ICE CDF)."""
    _configure_academic_style()
    env = report.baseline.dir_envelope
    dq = env.quality
    _dz, dy, dx = case.ct_course2.spacing_mm
    z_idx, _ = _find_best_slice_indices(case)

    body_ax = case.body_mask_course2[z_idx]
    r0, r1, c0, c1 = _body_crop_bounds(body_ax, pad=6)

    fig = plt.figure(figsize=(14.6, 4.9))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.12, 1.0, 1.0], wspace=0.40, top=0.84, bottom=0.18)

    # Panel (a): Axial Jacobian Map + DVF Quiver Overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ct_sl = case.ct_course2.data[z_idx, r0:r1, c0:c1]
    jac_sl = env.jacobian_map.data[z_idx, r0:r1, c0:c1]
    bsl = body_ax[r0:r1, c0:c1]

    ax1.imshow(ct_sl, cmap="gray", vmin=-250, vmax=450, aspect=dy / dx)
    j_masked = np.ma.masked_where(~bsl, jac_sl)
    im = ax1.imshow(j_masked, cmap="RdBu_r", vmin=0.75, vmax=1.25, alpha=0.72, aspect=dy / dx)

    # Subsampled 2D DVF vector quiver (uy, ux in pixels)
    step = 4
    yy_q, xx_q = np.mgrid[0 : (r1 - r0) : step, 0 : (c1 - c0) : step]
    uy_q = env.dvf_mm[1, z_idx, r0:r1:step, c0:c1:step] / dy
    ux_q = env.dvf_mm[2, z_idx, r0:r1:step, c0:c1:step] / dx
    b_q = bsl[::step, ::step]
    ax1.quiver(
        xx_q[b_q],
        yy_q[b_q],
        ux_q[b_q],
        -uy_q[b_q],  # negative y for top-down image axis
        color="#111111",
        scale=18,
        width=0.0045,
        alpha=0.80,
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.048, pad=0.04)
    cbar.set_label(r"Jacobian $J(\mathbf{x}) = \det(I + \nabla \mathbf{u})$", fontsize=8.0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Axial Jacobian Map & DVF Vectors (z={z_idx})", pad=8)
    _add_panel_label(ax1, "a", dark_bg=True)
    _add_scalebar(ax1, 30.0, dx, color="white")

    # Panel (b): Jacobian Determinant Distribution across Patient Body
    ax2 = fig.add_subplot(gs[0, 1])
    jac_vals = env.jacobian_map.data[case.body_mask_course2]
    ax2.hist(jac_vals, bins=48, color="#0072B2", alpha=0.82, edgecolor="white", linewidth=0.4, density=True)
    ax2.axvline(1.0, color="#111111", linestyle="--", linewidth=1.2, label="Volume-preserving (J = 1)")
    ax2.axvline(0.0, color="#C62828", linestyle="-", linewidth=1.6, label="Folding boundary (J = 0)")
    ax2.set_xlim(min(-0.05, dq.jacobian_min - 0.1), max(1.30, dq.jacobian_max + 0.08))
    ax2.set_xlabel(r"Jacobian Determinant $J(\mathbf{x})$", fontweight="bold")
    ax2.set_ylabel("Probability Density", fontweight="bold")
    ax2.set_title("3D Diffeomorphism Topology Check", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(
        0.04,
        0.74,
        f"Min J = {dq.jacobian_min:.3f}\nMed J = {dq.jacobian_median:.3f}\nMax J = {dq.jacobian_max:.3f}\nFolded = {dq.folded_voxel_count}",
        transform=ax2.transAxes,
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F4F6F8", "edgecolor": "#CCCCCC"},
    )
    ax2.legend(loc="upper right", fontsize=7.3)
    _add_panel_label(ax2, "b")

    # Panel (c): Inverse-Consistency Error (ICE) Empirical CDF
    ax3 = fig.add_subplot(gs[0, 2])
    ice_vals = np.sort(env.ice_map_mm.data[case.body_mask_course2])
    cdf = np.linspace(0.0, 100.0, ice_vals.size)
    # Downsample for fast crisp plotting
    stride = max(ice_vals.size // 1200, 1)
    ax3.plot(ice_vals[::stride], cdf[::stride], color="#009E73", linewidth=2.2, label="Empirical ICE CDF")
    ax3.axvline(2.0, color="#C62828", linestyle="--", linewidth=1.4, label="AAPM TG-132 Target (2.0 mm)")
    ax3.axhline(95.0, color="#777777", linestyle=":", linewidth=1.0)
    ax3.scatter([dq.ice_p95_mm], [95.0], color="#D55E00", s=55, zorder=6, label=f"p95 ICE = {dq.ice_p95_mm:.3f} mm")

    ax3.set_xlim(0.0, max(2.25, dq.ice_max_mm * 1.1))
    ax3.set_ylim(0.0, 103.0)
    ax3.set_xlabel("Inverse-Consistency Error — ICE (mm)", fontweight="bold")
    ax3.set_ylabel("Cumulative Tissue Voxels (%)", fontweight="bold")
    ax3.set_title("AAPM TG-132 Inverse-Consistency Audit", pad=8)
    ax3.grid(True, linestyle="--", alpha=0.25)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="lower right", fontsize=7.4)
    # Add clean inset zoom when SVF ICE is ultra-small (< 0.25 mm)
    if dq.ice_max_mm < 0.25:
        ax_in = ax3.inset_axes([0.26, 0.34, 0.56, 0.44])
        ax_in.plot(ice_vals[::stride] * 1000.0, cdf[::stride], color="#009E73", linewidth=1.8)
        ax_in.scatter([dq.ice_p95_mm * 1000.0], [95.0], color="#D55E00", s=28, zorder=6)
        ax_in.set_title("Sub-millimeter Zoom", fontsize=7.2, pad=3)
        ax_in.set_xlabel(r"ICE ($\mu$m)", fontsize=7.0)
        ax_in.set_ylabel("Voxels (%)", fontsize=7.0)
        ax_in.tick_params(labelsize=6.6)
        ax_in.grid(True, linestyle="--", alpha=0.25)
    _add_panel_label(ax3, "c")

    fig.suptitle(
        "AAPM Task Group 132 Deformable Image Registration Quality & Topology Forensics",
        fontsize=11.8,
        fontweight="bold",
        y=0.96,
    )
    fig.text(0.5, 0.025, PROVENANCE_FOOTER, ha="center", fontsize=7.2, color="#555555", style="italic")

    png_path = out_dir / "fig5_tg132_dir_forensics.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig5_tg132_dir_forensics.pdf")
    plt.close(fig)
    return png_path


def generate_all_publication_figures(
    case: ReIrradiationBenchmarkCase,
    report: AdversarialSweepReport,
    out_dir: str | Path,
) -> list[Path]:
    """Generate all 5 publication-grade academic figures (PNG + PDF) into `out_dir`."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return [
        plot_fig1_triplanar_biobudget(case, report, dest),
        plot_fig2_uncertainty_band_dvh(case, report, dest),
        plot_fig3_adversarial_tipping_phase(case, report, dest),
        plot_fig4_schedule_modality_matrix(report, dest),
        plot_fig5_tg132_dir_forensics(case, report, dest),
    ]
