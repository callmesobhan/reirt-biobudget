"""Editorial Clinical Monograph Visualizations for reirt-biobudget.

Generates 300-DPI journal-ready figures (PNG + vector PDF) in a Warm-Ivory Serif
Clinical Monograph aesthetic inspired by Lancet Oncology and Nature Medicine print plates:
- Warm archival ivory canvas (#FAF8F4), crisp white axes (#FFFFFF), taupe spines (#9C9488)
- Classical serif titles/headers with burgundy editorial rules (#7A1C24) & STIX math typesetting
- Clinical Crimson & Botanical Slate colorblind-safe palette with perceptually uniform colormaps
- Bold serif (a), (b), (c) badges, radiological CT scalebars, and full provenance footers
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
from matplotlib.lines import Line2D

from reirt_biobudget.budget_solver import compute_dvh_curve
from reirt_biobudget.cohort import CohortBenchmarkResult
from reirt_biobudget.harness.agent_orchestrator import AdversarialSweepReport
from reirt_biobudget.io_loaders import ReIrradiationBenchmarkCase
from reirt_biobudget.ir import (
    LegacyReconstructionAudit,
    OARNTCPSummary,
    SobolSensitivityResult,
    Volume3D,
)
from reirt_biobudget.legacy_reconstructor import LegacyReconstructionOutput

# ---------------------------------------------------------------------------
# Editorial Clinical Monograph Design Tokens
# ---------------------------------------------------------------------------
IVORY_CANVAS = "#FAF8F4"
AXES_WHITE = "#FFFFFF"
CHARCOAL_INK = "#1C1917"
MUTED_INK = "#5C544B"
TAUPE_SPINE = "#9C9488"
GRID_HAIRLINE = "#E6E0D6"
HEADER_BURGUNDY = "#7A1C24"

# Signature Colorblind-Safe Palette (Clinical Crimson & Botanical Slate)
CRIMSON = "#9E1B32"      # Deep Crimson (SpinalCord / Violation)
NAVY = "#1B3B6F"         # Imperial Navy (Brainstem / Primary Quantitative)
FOREST = "#2D6A4F"       # Botanical Forest (Bone_Mandible / Safe)
PLUM = "#7B3E6B"         # Mulberry Plum (Parotid_L)
TERRACOTTA = "#C96F28"   # Warm Terracotta (Parotid_R / Caution)
SLATE = "#457B9D"        # Steel Slate (Larynx)
INDIGO = "#2E294E"       # Midnight Indigo (Esophagus)

SAFE_FILL = "#EBF3EE"      # Warm parchment-sage
CAUTION_FILL = "#FDF3E3"   # Warm cream-amber
PTV_CONTOUR = "#F4C95D"    # Archival gold dashed salvage PTV contour on CT
PTV_LEGEND = "#B88414"     # Darker archival gold for crisp legend contrast on ivory

OAR_COLORS: dict[str, str] = {
    "SpinalCord": CRIMSON,
    "Brainstem": NAVY,
    "Bone_Mandible": FOREST,
    "Parotid_L": PLUM,
    "Parotid_R": TERRACOTTA,
    "Larynx": SLATE,
    "Esophagus": INDIGO,
}

# Custom monograph diverging colormap for 2D safety phase diagram (Forest -> Parchment -> Crimson)
SAFETY_PHASE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "monograph_safety_phase",
    [
        (0.00, "#21523D"),
        (0.22, "#4F8A6E"),
        (0.42, "#DCEAE1"),
        (0.50, "#FDF3E3"),
        (0.66, "#DE8F47"),
        (0.84, "#9E1B32"),
        (1.00, "#590D1A"),
    ],
)

PROVENANCE_FOOTER = (
    "Source data: OpenKBP public clinical dataset (pt_1; Babier et al., Med Phys 2021, "
    "DOI: 10.1002/mp.14845, CC BY 4.0) with documented SVF longitudinal benchmark. "
    "Research tool — not a medical device."
)


def _configure_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.8,
            "text.color": CHARCOAL_INK,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.titlecolor": CHARCOAL_INK,
            "axes.labelsize": 8.8,
            "axes.labelcolor": CHARCOAL_INK,
            "axes.edgecolor": TAUPE_SPINE,
            "axes.linewidth": 0.85,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.color": CHARCOAL_INK,
            "ytick.color": CHARCOAL_INK,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "legend.fontsize": 7.7,
            "legend.frameon": True,
            "legend.facecolor": IVORY_CANVAS,
            "legend.framealpha": 0.95,
            "legend.edgecolor": TAUPE_SPINE,
            "figure.facecolor": IVORY_CANVAS,
            "axes.facecolor": AXES_WHITE,
            "savefig.facecolor": IVORY_CANVAS,
            "savefig.edgecolor": "none",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "pdf.fonttype": 42,
        }
    )


def _style_editorial_axes(
    ax: plt.Axes,
    *,
    horizontal_grid: bool = True,
    keep_all_spines: bool = False,
) -> None:
    """Apply Warm-Ivory Monograph spine & horizontal editorial hairline styling."""
    ax.set_facecolor(AXES_WHITE)
    ax.set_axisbelow(True)
    if keep_all_spines:
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(TAUPE_SPINE)
            ax.spines[side].set_linewidth(0.85)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(TAUPE_SPINE)
            ax.spines[side].set_linewidth(0.85)

    ax.grid(False, axis="both")
    if horizontal_grid:
        ax.grid(True, axis="y", color=GRID_HAIRLINE, linestyle="-", linewidth=0.6, alpha=1.0)


def _style_ct_axes(ax: plt.Axes) -> None:
    """Frame clinical CT slices cleanly with subtle taupe borders."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(TAUPE_SPINE)
        ax.spines[side].set_linewidth(0.85)


def _style_colorbar(cbar: plt.Colorbar, label: str, fontsize: float = 8.0) -> None:
    """Apply monograph typography and taupe outline to colorbars."""
    cbar.set_label(label, fontsize=fontsize, color=CHARCOAL_INK)
    cbar.ax.tick_params(labelsize=7.4, colors=CHARCOAL_INK, width=0.7)
    cbar.outline.set_edgecolor(TAUPE_SPINE)
    cbar.outline.set_linewidth(0.8)


def _add_editorial_chrome(
    fig: plt.Figure,
    title: str,
    footer: str = PROVENANCE_FOOTER,
    *,
    y_title: float = 0.975,
    y_rule: float = 0.930,
    y_footer: float = 0.018,
    x_left: float = 0.04,
    x_right: float = 0.96,
    title_size: float = 11.8,
) -> None:
    """Render serif figure suptitle, crisp burgundy accent rule, and provenance footer."""
    fig.suptitle(
        title,
        fontfamily="serif",
        fontsize=title_size,
        fontweight="bold",
        color=CHARCOAL_INK,
        y=y_title,
    )
    fig.add_artist(
        Line2D(
            [x_left, x_right],
            [y_rule, y_rule],
            transform=fig.transFigure,
            color=HEADER_BURGUNDY,
            linewidth=2.0,
            solid_capstyle="butt",
        )
    )
    fig.add_artist(
        Line2D(
            [x_left, x_right],
            [y_footer + 0.028, y_footer + 0.028],
            transform=fig.transFigure,
            color=GRID_HAIRLINE,
            linewidth=0.75,
            solid_capstyle="butt",
        )
    )
    fig.text(
        0.5,
        y_footer,
        footer,
        ha="center",
        va="center",
        fontfamily="serif",
        fontsize=7.2,
        color=MUTED_INK,
        style="italic",
    )


def _add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    dark_bg: bool = False,
    loc: str = "upper left",
) -> None:
    """Draw bold serif (a), (b), (c) inside editorial ivory/burgundy badge or dark CT pill."""
    clean = label.strip().strip("()")
    text = f"({clean})"
    if loc == "upper right":
        x, y, ha, va = 0.978, 0.968, "right", "top"
    elif loc == "lower right":
        x, y, ha, va = 0.978, 0.042, "right", "bottom"
    else:
        x, y, ha, va = 0.022, 0.968, "left", "top"

    if dark_bg:
        bbox = {
            "boxstyle": "round,pad=0.22,rounding_size=0.18",
            "facecolor": (0.26, 0.06, 0.09, 0.80),
            "edgecolor": "#D9A7AB",
            "linewidth": 0.75,
        }
        color = IVORY_CANVAS
    else:
        bbox = {
            "boxstyle": "round,pad=0.22,rounding_size=0.18",
            "facecolor": IVORY_CANVAS,
            "edgecolor": HEADER_BURGUNDY,
            "linewidth": 0.85,
        }
        color = HEADER_BURGUNDY

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontfamily="serif",
        fontsize=9.4,
        fontweight="bold",
        va=va,
        ha=ha,
        color=color,
        bbox=bbox,
        zorder=20,
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


def _add_scalebar(
    ax: plt.Axes,
    length_mm: float,
    pixel_mm: float,
    color: str = IVORY_CANVAS,
) -> None:
    """Draw a crisp physical millimeter scalebar with high-contrast monograph backing."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bar_px = length_mm / pixel_mm
    x_end = xlim[1] - 0.06 * (xlim[1] - xlim[0])
    x_start = x_end - bar_px
    y_pos = ylim[0] + 0.075 * (ylim[1] - ylim[0])  # ylim is inverted for imshow
    ax.plot([x_start, x_end], [y_pos, y_pos], color="#1C1917", linewidth=4.0, solid_capstyle="butt")
    ax.plot([x_start, x_end], [y_pos, y_pos], color=color, linewidth=2.0, solid_capstyle="butt")
    ax.text(
        0.5 * (x_start + x_end),
        y_pos - 0.036 * (ylim[1] - ylim[0]),
        f"{length_mm:.0f} mm",
        color=color,
        fontsize=7.2,
        fontweight="bold",
        ha="center",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": (0.11, 0.09, 0.08, 0.72),
            "edgecolor": "none",
        },
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

    fig = plt.figure(figsize=(14.8, 8.1))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        wspace=0.18,
        hspace=0.22,
        left=0.06,
        right=0.97,
        top=0.83,
        bottom=0.11,
    )

    col_titles = [
        "Follow-Up CT & TG-263 Anatomy",
        "Prior Effective EQD2 (Gy)",
        r"DIR Spread ($p_{95} - p_{05}$, Gy)",
        r"Remaining Budget $p_{95}$ (Gy)",
    ]

    overlays = [
        None,
        (
            prior_eqd2,
            "magma",
            5.0,
            max(float(np.percentile(prior_eqd2[case.body_mask_course2], 99.5)), 60.0),
            "Prior EQD2 (Gy)",
        ),
        (
            spread_eqd2,
            "YlOrRd",
            0.5,
            max(float(np.percentile(spread_eqd2[case.body_mask_course2], 99.5)), 8.0),
            "90% DIR Spread (Gy)",
        ),
        (
            rem_phys,
            "cividis",
            0.0,
            max(float(np.percentile(rem_phys[case.body_mask_course2], 98)), 25.0),
            "Remaining Physical Ceiling (Gy)",
        ),
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
                        colors=[OAR_COLORS.get(oar_name, IVORY_CANVAS)],
                        linewidths=1.45,
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
                    colors=[PTV_CONTOUR],
                    linewidths=1.6,
                    linestyles="--",
                )

            if row_i == 0:
                ax.set_title(col_titles[col_i], fontfamily="serif", fontsize=9.8, pad=8)
            if col_i == 0:
                orient = "R → L (Axial)" if plane == "Axial" else "A → P (Sagittal)"
                ax.set_ylabel(
                    f"{plane} Slice #{sl_idx}\n({orient})",
                    fontfamily="serif",
                    fontweight="bold",
                    fontsize=8.6,
                    color=CHARCOAL_INK,
                )
            _style_ct_axes(ax)
            _add_panel_label(ax, labels[col_i], dark_bg=True)
            if col_i == 0:
                _add_scalebar(ax, 30.0, px_mm)

            if row_i == 1 and im_ov is not None:
                cbar = fig.colorbar(
                    im_ov,
                    ax=ax,
                    orientation="horizontal",
                    fraction=0.056,
                    pad=0.06,
                )
                _style_colorbar(cbar, ov[4], fontsize=7.9)

    legend_handles = [
        mpatches.Patch(
            facecolor="none",
            edgecolor=PTV_LEGEND,
            linestyle="--",
            linewidth=1.6,
            label="Recurrent Salvage PTV",
        ),
    ]
    for oar_name in ["SpinalCord", "Brainstem", "Bone_Mandible", "Parotid_L", "Parotid_R"]:
        if oar_name in case.oar_masks_course2:
            legend_handles.append(
                mpatches.Patch(
                    facecolor="none",
                    edgecolor=OAR_COLORS[oar_name],
                    linewidth=1.7,
                    label=oar_name,
                )
            )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.515, 0.922),
        ncol=len(legend_handles),
        frameon=True,
        facecolor=AXES_WHITE,
        edgecolor=TAUPE_SPINE,
        fontsize=8.0,
        columnspacing=1.3,
    )
    _add_editorial_chrome(
        fig,
        f"3D Biophysical Dose Accumulation, DIR-Uncertainty Envelope & Remaining Physical Budget ({case.case_id})",
        y_title=0.985,
        y_rule=0.942,
        y_footer=0.016,
        x_left=0.04,
        x_right=0.97,
        title_size=11.9,
    )

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

    fig = plt.figure(figsize=(14.0, 5.8))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.18, 1.04],
        wspace=0.28,
        left=0.06,
        right=0.97,
        top=0.79,
        bottom=0.16,
    )

    # Panel (a): Cumulative EQD2 DVHs with 90% DIR uncertainty ribbons
    ax1 = fig.add_subplot(gs[0, 0])
    for s in summaries:
        oar = s.tg263_name
        env = out.oar_eqd2_envelopes.get(oar)
        if env is None:
            continue
        color = OAR_COLORS.get(oar, CHARCOAL_INK)
        dvh_p05 = compute_dvh_curve(env["cum_p05"], dose_grid)
        dvh_nom = compute_dvh_curve(env["cum_nominal"], dose_grid)
        dvh_p95 = compute_dvh_curve(env["cum_p95"], dose_grid)
        dvh_worst = compute_dvh_curve(env["cum_worst"], dose_grid)

        ax1.fill_between(dose_grid, dvh_p05, dvh_p95, color=color, alpha=0.18, linewidth=0)
        ax1.plot(dose_grid, dvh_nom, color=color, linewidth=2.0, label=f"{oar} ({s.metric_name})")
        ax1.plot(dose_grid, dvh_worst, color=color, linewidth=0.95, linestyle=":", alpha=0.9)

        if oar in ("SpinalCord", "Brainstem"):
            ax1.axvline(s.eqd2_limit_gy, color=color, linestyle="--", linewidth=1.05, alpha=0.72)
            ax1.text(
                s.eqd2_limit_gy + 1.0,
                97.0 if oar == "SpinalCord" else 83.0,
                f"{oar}\nLimit ({s.eqd2_limit_gy:.0f} Gy)",
                color=color,
                fontsize=7.3,
                fontweight="bold",
                va="top",
            )

    ax1.set_xlim(0, 95)
    ax1.set_ylim(0, 113)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlabel("Cumulative Effective EQD2 (Gy)", fontweight="bold")
    ax1.set_ylabel("Relative OAR Volume (%)", fontweight="bold")
    ax1.set_title(
        "Cumulative EQD2 DVHs with 90% DIR Uncertainty Envelopes",
        fontfamily="serif",
        pad=10,
    )
    _style_editorial_axes(ax1, horizontal_grid=True)

    handles, _labels = ax1.get_legend_handles_labels()
    handles.append(
        mpatches.Patch(facecolor=MUTED_INK, alpha=0.22, label=r"90% DIR band $[p_{05}, p_{95}]$")
    )
    handles.append(
        Line2D([0], [0], color=CHARCOAL_INK, linestyle=":", linewidth=1.1, label="Ensemble worst-case")
    )
    ax1.legend(handles=handles, loc="lower left", ncol=2, fontsize=7.3)
    _add_panel_label(ax1, "a")

    # Panel (b): OAR Prior vs. Cumulative EQD2 Dot-Whisker against Caution & Limit
    ax2 = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(summaries))
    for i, s in enumerate(summaries):
        ax2.barh(i, s.eqd2_caution_gy, height=0.52, color=SAFE_FILL, edgecolor="none", zorder=1)
        ax2.barh(
            i,
            s.eqd2_limit_gy - s.eqd2_caution_gy,
            left=s.eqd2_caution_gy,
            height=0.52,
            color=CAUTION_FILL,
            edgecolor="none",
            zorder=1,
        )
        ax2.plot(
            [s.eqd2_limit_gy, s.eqd2_limit_gy],
            [i - 0.30, i + 0.30],
            color=CRIMSON,
            linewidth=2.1,
            zorder=4,
        )

        # Prior Effective [p05, p95] whisker
        ax2.plot(
            [s.prior_eqd2_effective_p05_gy, s.prior_eqd2_effective_p95_gy],
            [i + 0.11, i + 0.11],
            color=MUTED_INK,
            linewidth=2.3,
            zorder=5,
        )
        ax2.scatter(
            [s.prior_eqd2_effective_nominal_gy],
            [i + 0.11],
            color=CHARCOAL_INK,
            s=34,
            zorder=6,
            label="Prior Eff. EQD2 (Nom, [p05–p95])" if i == 0 else None,
        )

        if s.cumulative_eqd2_nominal_gy is not None and s.cumulative_eqd2_p95_gy is not None:
            v_col = (
                CRIMSON
                if s.verdict.value == "VIOLATION"
                else (TERRACOTTA if s.verdict.value == "CAUTION" else FOREST)
            )
            ax2.plot(
                [s.cumulative_eqd2_nominal_gy, s.cumulative_eqd2_p95_gy],
                [i - 0.12, i - 0.12],
                color=v_col,
                linewidth=2.7,
                zorder=5,
            )
            ax2.scatter(
                [s.cumulative_eqd2_p95_gy],
                [i - 0.12],
                color=v_col,
                marker="D",
                s=40,
                zorder=7,
                label="Cumulative p95 EQD2 (with Course 2)" if i == 0 else None,
            )
            ax2.text(
                max(s.cumulative_eqd2_p95_gy, s.eqd2_limit_gy) + 2.0,
                i,
                f"{s.verdict.value}\nRem: {s.remaining_physical_p95_gy:.1f} Gy",
                va="center",
                fontsize=7.3,
                fontweight="bold",
                color=v_col,
            )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{s.tg263_name}\n({s.metric_name})" for s in summaries], fontsize=8.0)
    ax2.set_ylim(len(summaries) - 0.45, -0.88)
    max_x = max(max(s.eqd2_limit_gy, s.cumulative_eqd2_p95_gy or 0.0) for s in summaries) + 28.0
    ax2.set_xlim(0, max_x)
    ax2.set_xlabel("Equivalent Dose in 2 Gy Fractions — EQD2 (Gy)", fontweight="bold")
    ax2.set_title(
        "OAR Tolerance Headroom & Remaining Physical Budget",
        fontfamily="serif",
        pad=28,
    )
    _style_editorial_axes(ax2, horizontal_grid=True)

    p_handles = [
        mpatches.Patch(facecolor=SAFE_FILL, edgecolor=TAUPE_SPINE, linewidth=0.5, label="Safe"),
        mpatches.Patch(
            facecolor=CAUTION_FILL, edgecolor=TAUPE_SPINE, linewidth=0.5, label="Caution"
        ),
        Line2D([0], [0], color=CRIMSON, linewidth=2.1, label="Limit"),
        Line2D(
            [0],
            [0],
            color=MUTED_INK,
            marker="o",
            markerfacecolor=CHARCOAL_INK,
            markeredgecolor=CHARCOAL_INK,
            linewidth=2.0,
            markersize=4.5,
            label=r"Prior $[p_{05}$–$p_{95}]$",
        ),
        Line2D(
            [0],
            [0],
            color=TERRACOTTA,
            marker="D",
            linewidth=2.0,
            markersize=4.5,
            label=r"Cum. $p_{95}$",
        ),
    ]
    ax2.legend(
        handles=p_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        fontsize=6.8,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    _add_panel_label(ax2, "b")

    _add_editorial_chrome(
        fig,
        f"Biophysical Cumulative EQD2 DVHs & OAR Safety Headroom ({out.result.planned_course.label})",
        y_title=0.975,
        y_rule=0.918,
        y_footer=0.022,
        x_left=0.04,
        x_right=0.97,
        title_size=11.8,
    )

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

    fig = plt.figure(figsize=(14.2, 6.1))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.16, 1.0],
        wspace=0.25,
        left=0.09,
        right=0.96,
        top=0.75,
        bottom=0.15,
    )

    # Panel (a): Stacked Horizontal Sensitivity Decomposition per OAR
    ax1 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(tps))
    sw_rec = np.array([t.swing_recovery_gy for t in tps])
    sw_tre = np.array([t.swing_dir_tre_gy for t in tps])
    sw_ab = np.array([t.swing_alpha_beta_gy for t in tps])
    sw_mod = np.array([t.swing_model_regime_gy for t in tps])

    h = 0.54
    ax1.barh(
        y,
        sw_rec,
        height=h,
        color=NAVY,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label="Tissue Recovery Credit (0% to max)",
    )
    ax1.barh(
        y,
        sw_tre,
        left=sw_rec,
        height=h,
        color=CRIMSON,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label=r"DIR $\sigma_{\mathrm{TRE}}$ Uncertainty (0–4 mm)",
    )
    ax1.barh(
        y,
        sw_ab,
        left=sw_rec + sw_tre,
        height=h,
        color=FOREST,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label=r"Radiobiology $\alpha/\beta$ Range",
    )
    ax1.barh(
        y,
        sw_mod,
        left=sw_rec + sw_tre + sw_ab,
        height=h,
        color=PLUM,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label="Model Regime (LQ vs. LQL)",
    )

    totals = sw_rec + sw_tre + sw_ab + sw_mod
    short_driver_map = {
        "Tissue Recovery Credit": "Recovery",
        "DIR Registration TRE (0-4 mm)": "DIR TRE",
        "Radiobiology alpha/beta Uncertainty": r"$\alpha/\beta$",
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
            totals[i] + 0.45,
            i,
            f"Top: {short_d} ({top_val:.1f} Gy){flag}",
            va="center",
            fontsize=7.4,
            fontweight="bold" if t.flips_to_violation else "normal",
            color=CRIMSON if t.flips_to_violation else CHARCOAL_INK,
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{t.oar_name}\n({t.metric_name})" for t in tps], fontsize=8.0)
    ax1.set_ylim(len(tps) - 0.45, -0.88)
    ax1.set_xlim(0, max(float(np.max(totals)) * 1.66, 18.0))
    ax1.set_xlabel("Cumulative EQD2 Sensitivity Swing — Sum of Factor Ranges (Gy)", fontweight="bold")
    ax1.set_title("Adversarial Assumption Sensitivity Decomposition", fontfamily="serif", pad=44)
    _style_editorial_axes(ax1, horizontal_grid=True)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
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
    cf = ax2.contourf(R, T, Z, levels=20, cmap=SAFETY_PHASE_CMAP, norm=norm, alpha=0.92)
    cbar = fig.colorbar(cf, ax=ax2, pad=0.03)
    _style_colorbar(cbar, f"{pd.oar_name} Cumulative $p_{{95}}$ EQD2 (Gy)", fontsize=8.2)

    if float(np.min(Z)) < pd.eqd2_caution_gy < float(np.max(Z)):
        cs_c = ax2.contour(
            R,
            T,
            Z,
            levels=[pd.eqd2_caution_gy],
            colors=["#8C430E"],
            linewidths=2.0,
            linestyles="--",
        )
        ax2.clabel(cs_c, fmt=f"Caution ({pd.eqd2_caution_gy:.0f} Gy)", fontsize=7.8)
    if float(np.min(Z)) < pd.eqd2_limit_gy < float(np.max(Z)):
        cs_l = ax2.contour(
            R,
            T,
            Z,
            levels=[pd.eqd2_limit_gy],
            colors=[HEADER_BURGUNDY],
            linewidths=2.4,
        )
        ax2.clabel(cs_l, fmt=f"Limit ({pd.eqd2_limit_gy:.0f} Gy)", fontsize=8.0)

    base_s = next(s for s in report.baseline.result.oar_summaries if s.tg263_name == pd.oar_name)
    base_r_pct = base_s.recovery_fraction_applied * 100.0
    base_tre = report.baseline.result.dir_quality.ensemble_tre_sigma_mm
    ax2.scatter(
        [base_r_pct],
        [base_tre],
        color=IVORY_CANVAS,
        edgecolor=CHARCOAL_INK,
        s=105,
        marker="*",
        linewidth=1.2,
        zorder=8,
        label=f"Baseline ({base_r_pct:.1f}% rec., {base_tre:.1f} mm TRE)",
    )
    ax2.set_xlabel("Assumed Prior OAR Recovery Credit (%)", fontweight="bold")
    ax2.set_ylabel(r"DIR Uncertainty $\sigma_{\mathrm{TRE}}$ (mm)", fontweight="bold")
    ax2.set_title(f"{pd.oar_name} Clinical Safety Phase Diagram", fontfamily="serif", pad=10)
    _style_editorial_axes(ax2, horizontal_grid=False, keep_all_spines=True)
    ax2.grid(True, which="major", color=CHARCOAL_INK, linestyle=":", linewidth=0.45, alpha=0.22)
    ax2.legend(loc="upper right", fontsize=7.5)
    _add_panel_label(ax2, "b")

    _add_editorial_chrome(
        fig,
        "Frontier Agent Harness: Adversarial Failure-Mode Discovery & Clinical Decision Boundaries",
        y_title=0.975,
        y_rule=0.918,
        y_footer=0.022,
        x_left=0.04,
        x_right=0.96,
        title_size=11.8,
    )

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
    key_oars = [
        o for o in ["SpinalCord", "Brainstem", "Bone_Mandible"] if o in rows[0].oar_remaining_physical_p95_gy
    ]

    fig = plt.figure(figsize=(14.6, 5.9))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.22, 1.05],
        wspace=0.40,
        left=0.06,
        right=0.97,
        top=0.79,
        bottom=0.19,
    )

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
            color=OAR_COLORS.get(oar, MUTED_INK),
            label=oar,
            edgecolor=AXES_WHITE,
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
                color=CHARCOAL_INK,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=7.5, ha="center")
    ax1.set_ylim(0, max_bar * 1.24)
    ax1.set_ylabel(r"Safe Remaining Physical Dose Ceiling $p_{95}$ (Gy)", fontweight="bold")
    ax1.set_title("Exact Inverted Course-2 Physical Dose Ceiling by Schedule", fontfamily="serif", pad=26)
    _style_editorial_axes(ax1, horizontal_grid=True)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7.8, ncol=3)
    _add_panel_label(ax1, "a")

    # Panel (b): Limiting Serial OAR EQD2 Margin
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(rows))
    margins = np.array([r.limiting_oar_margin_eqd2_gy for r in rows])
    bar_colors = [
        CRIMSON if r.verdict.value == "VIOLATION" else (TERRACOTTA if r.verdict.value == "CAUTION" else FOREST)
        for r in rows
    ]
    ax2.barh(
        y,
        margins,
        height=0.52,
        color=bar_colors,
        alpha=0.90,
        edgecolor=CHARCOAL_INK,
        linewidth=0.55,
    )
    ax2.axvline(0.0, color=CHARCOAL_INK, linewidth=1.25)

    x_lo = min(float(np.min(margins)) - 6.0, -8.0)
    x_hi = max(float(np.max(margins)) + 28.0, 28.0)
    for i, r in enumerate(rows):
        m_val = r.limiting_oar_margin_eqd2_gy
        txt_x = max(m_val, 0.0) + 0.9
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
    ax2.set_ylim(len(rows) - 0.45, -0.88)
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_xlabel(r"Serial OAR Margin: $\mathrm{Limit} - \mathrm{Cum.}\ p_{95}\ \mathrm{EQD2}$ (Gy)", fontweight="bold")
    ax2.set_title("Critical Serial-OAR Safety Margin & Schedule Triage", fontfamily="serif", pad=26)
    _style_editorial_axes(ax2, horizontal_grid=True)
    _add_panel_label(ax2, "b")

    _add_editorial_chrome(
        fig,
        "Multi-Schedule & Multi-Modality Co-Design (Photon LQ / LQL / BID Lea–Catcheside vs. Variable-RBE Proton)",
        y_title=0.975,
        y_rule=0.918,
        y_footer=0.022,
        x_left=0.04,
        x_right=0.97,
        title_size=11.4,
    )

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

    fig = plt.figure(figsize=(14.8, 5.4))
    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.12, 1.0, 1.0],
        wspace=0.38,
        left=0.045,
        right=0.97,
        top=0.80,
        bottom=0.18,
    )

    # Panel (a): Axial Jacobian Map + DVF Quiver Overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ct_sl = case.ct_course2.data[z_idx, r0:r1, c0:c1]
    jac_sl = env.jacobian_map.data[z_idx, r0:r1, c0:c1]
    bsl = body_ax[r0:r1, c0:c1]

    ax1.imshow(ct_sl, cmap="gray", vmin=-250, vmax=450, aspect=dy / dx)
    j_masked = np.ma.masked_where(~bsl, jac_sl)
    im = ax1.imshow(j_masked, cmap="coolwarm", vmin=0.75, vmax=1.25, alpha=0.74, aspect=dy / dx)

    step = 4
    yy_q, xx_q = np.mgrid[0 : (r1 - r0) : step, 0 : (c1 - c0) : step]
    uy_q = env.dvf_mm[1, z_idx, r0:r1:step, c0:c1:step] / dy
    ux_q = env.dvf_mm[2, z_idx, r0:r1:step, c0:c1:step] / dx
    b_q = bsl[::step, ::step]
    ax1.quiver(
        xx_q[b_q],
        yy_q[b_q],
        ux_q[b_q],
        -uy_q[b_q],
        color=CHARCOAL_INK,
        scale=18,
        width=0.0045,
        alpha=0.82,
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.048, pad=0.04)
    _style_colorbar(cbar, r"Jacobian $J(\mathbf{x}) = \det(I + \nabla \mathbf{u})$", fontsize=8.0)
    _style_ct_axes(ax1)
    ax1.set_title(f"Axial Jacobian Map & DVF Vectors ($z = {z_idx}$)", fontfamily="serif", pad=9)
    _add_panel_label(ax1, "a", dark_bg=True)
    _add_scalebar(ax1, 30.0, dx)

    # Panel (b): Jacobian Determinant Distribution across Patient Body
    ax2 = fig.add_subplot(gs[0, 1])
    jac_vals = env.jacobian_map.data[case.body_mask_course2]
    ax2.hist(
        jac_vals,
        bins=48,
        color=NAVY,
        alpha=0.84,
        edgecolor=IVORY_CANVAS,
        linewidth=0.45,
        density=True,
    )
    ax2.axvline(1.0, color=CHARCOAL_INK, linestyle="--", linewidth=1.2, label="Volume-preserving ($J = 1$)")
    ax2.axvline(0.0, color=CRIMSON, linestyle="-", linewidth=1.7, label="Folding boundary ($J = 0$)")
    ax2.set_xlim(min(-0.05, dq.jacobian_min - 0.1), max(1.30, dq.jacobian_max + 0.08))
    ax2.set_xlabel(r"Jacobian Determinant $J(\mathbf{x})$", fontweight="bold")
    ax2.set_ylabel("Probability Density", fontweight="bold")
    ax2.set_title("3D Diffeomorphism Topology Check", fontfamily="serif", pad=9)
    _style_editorial_axes(ax2, horizontal_grid=True)
    ax2.text(
        0.04,
        0.79,
        f"Min $J$ = {dq.jacobian_min:.3f}\nMed $J$ = {dq.jacobian_median:.3f}\n"
        f"Max $J$ = {dq.jacobian_max:.3f}\nFolded = {dq.folded_voxel_count}",
        transform=ax2.transAxes,
        fontsize=7.6,
        va="top",
        color=CHARCOAL_INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": IVORY_CANVAS,
            "edgecolor": TAUPE_SPINE,
            "linewidth": 0.8,
        },
    )
    ax2.legend(loc="upper right", fontsize=7.2)
    _add_panel_label(ax2, "b")

    # Panel (c): Inverse-Consistency Error (ICE) Empirical CDF
    ax3 = fig.add_subplot(gs[0, 2])
    ice_vals = np.sort(env.ice_map_mm.data[case.body_mask_course2])
    cdf = np.linspace(0.0, 100.0, ice_vals.size)
    stride = max(ice_vals.size // 1200, 1)
    ax3.plot(ice_vals[::stride], cdf[::stride], color=FOREST, linewidth=2.2, label="Empirical ICE CDF")
    ax3.axvline(2.0, color=CRIMSON, linestyle="--", linewidth=1.4, label="AAPM TG-132 Target (2.0 mm)")
    ax3.axhline(95.0, color=TAUPE_SPINE, linestyle=":", linewidth=1.0)
    ax3.scatter(
        [dq.ice_p95_mm],
        [95.0],
        color=TERRACOTTA,
        edgecolor=CHARCOAL_INK,
        linewidth=0.6,
        s=55,
        zorder=6,
        label=f"$p_{{95}}$ ICE = {dq.ice_p95_mm:.3f} mm",
    )

    ax3.set_xlim(0.0, max(2.25, dq.ice_max_mm * 1.1))
    ax3.set_ylim(0.0, 113.0)
    ax3.set_yticks([0, 20, 40, 60, 80, 100])
    ax3.set_xlabel("Inverse-Consistency Error — ICE (mm)", fontweight="bold")
    ax3.set_ylabel("Cumulative Tissue Voxels (%)", fontweight="bold")
    ax3.set_title("AAPM TG-132 Inverse-Consistency Audit", fontfamily="serif", pad=9)
    _style_editorial_axes(ax3, horizontal_grid=True)
    ax3.legend(loc="lower right", fontsize=7.3)
    if dq.ice_max_mm < 0.25:
        ax_in = ax3.inset_axes([0.27, 0.34, 0.56, 0.43])
        ax_in.plot(ice_vals[::stride] * 1000.0, cdf[::stride], color=FOREST, linewidth=1.8)
        ax_in.scatter(
            [dq.ice_p95_mm * 1000.0],
            [95.0],
            color=TERRACOTTA,
            edgecolor=CHARCOAL_INK,
            linewidth=0.5,
            s=30,
            zorder=6,
        )
        ax_in.set_title("Sub-millimeter Zoom", fontfamily="serif", fontsize=7.4, pad=3)
        ax_in.set_xlabel(r"ICE ($\mu$m)", fontsize=7.0)
        ax_in.set_ylabel("Voxels (%)", fontsize=7.0)
        ax_in.tick_params(labelsize=6.5)
        _style_editorial_axes(ax_in, horizontal_grid=True, keep_all_spines=True)
    _add_panel_label(ax3, "c")

    _add_editorial_chrome(
        fig,
        "AAPM Task Group 132 Deformable Image Registration Quality & Topology Forensics",
        y_title=0.975,
        y_rule=0.915,
        y_footer=0.024,
        x_left=0.04,
        x_right=0.97,
        title_size=11.8,
    )

    png_path = out_dir / "fig5_tg132_dir_forensics.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig5_tg132_dir_forensics.pdf")
    plt.close(fig)
    return png_path


def plot_fig6_cohort_landmark_validation(
    cohort: CohortBenchmarkResult,
    out_dir: Path,
) -> Path:
    """Figure 6: Multi-Patient OpenKBP Cohort TRE Validation, 90% Coverage & K=2 Headroom."""
    _configure_academic_style()
    pats = cohort.patients
    pat_labels = [p.patient_id.upper() for p in pats]

    fig = plt.figure(figsize=(15.0, 5.8))
    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.05, 1.0, 1.25],
        wspace=0.34,
        left=0.055,
        right=0.975,
        top=0.73,
        bottom=0.17,
    )

    # Panel (a): Pre-DIR vs Post-DIR Anatomical Landmark TRE (mm)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(pats))
    w = 0.34
    pre_mean = [p.landmark_validation.pre_dir_tre_mean_mm for p in pats]
    pre_p95 = [p.landmark_validation.pre_dir_tre_p95_mm for p in pats]
    post_p95 = [p.landmark_validation.post_dir_tre_p95_mm for p in pats]

    b1 = ax1.bar(
        x - w / 2,
        pre_mean,
        width=w * 0.9,
        color=CRIMSON,
        alpha=0.88,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label="Pre-DIR Mean TRE",
    )
    b2 = ax1.bar(
        x + w / 2,
        pre_p95,
        width=w * 0.9,
        color=TERRACOTTA,
        alpha=0.88,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label=r"Pre-DIR $p_{95}$ TRE",
    )
    ax1.bar(
        x + w / 2,
        post_p95,
        width=w * 0.9,
        color=FOREST,
        edgecolor=AXES_WHITE,
        linewidth=0.4,
        label=r"Post-DIR $p_{95}$ TRE",
    )
    ax1.axhline(2.0, color=CHARCOAL_INK, linestyle="--", linewidth=1.2, label="AAPM TG-132 (2.0 mm)")

    for bar_obj, val in zip(b1, pre_mean):
        ax1.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            val + 0.12,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=CHARCOAL_INK,
        )
    for bar_obj, val, pval in zip(b2, pre_p95, post_p95):
        ax1.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            val + 0.12,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=CHARCOAL_INK,
        )
        ax1.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            0.22,
            f"<{max(pval, 0.01):.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color=FOREST,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{lbl}\n(n={p.landmark_validation.n_landmarks} LM)" for lbl, p in zip(pat_labels, pats)],
        fontsize=7.8,
    )
    ax1.set_ylim(0, max(max(pre_p95) * 1.28, 4.6))
    ax1.set_ylabel("Anatomical Landmark TRE (mm)", fontweight="bold")
    ax1.set_title("Pre- vs. Post-DIR Landmark TRE", fontfamily="serif", pad=40)
    _style_editorial_axes(ax1, horizontal_grid=True)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.7, columnspacing=0.8)
    _add_panel_label(ax1, "a")

    # Panel (b): Empirical 90% DIR Envelope Calibration across Patients
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(pats))
    covs = [p.landmark_validation.empirical_90pct_envelope_coverage_pct for p in pats]
    ax2.axvspan(85.0, 95.0, color=SAFE_FILL, alpha=0.95, zorder=1, label="Target Calibration [85–95%]")
    ax2.axvline(90.0, color=FOREST, linestyle="--", linewidth=1.4, zorder=2, label="Nominal 90% Target")
    ax2.barh(y, covs, height=0.48, color=NAVY, alpha=0.90, edgecolor=AXES_WHITE, linewidth=0.5, zorder=3)
    for i, (cval, p) in enumerate(zip(covs, pats)):
        ax2.text(
            min(cval + 1.2, 101.0),
            i,
            f"{cval:.1f}%\n({p.landmark_validation.gradient_voxels_tested:,} vox)",
            va="center",
            ha="left",
            fontsize=7.3,
            fontweight="bold",
            color=NAVY,
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels(pat_labels, fontsize=8.2, fontweight="bold")
    ax2.set_ylim(len(pats) - 0.45, -0.78)
    ax2.set_xlim(0, 126.0)
    ax2.set_xlabel("Empirical 90% Envelope Coverage (%)", fontweight="bold")
    ax2.set_title("DIR-Envelope Calibration Stress Test", fontfamily="serif", pad=40)
    _style_editorial_axes(ax2, horizontal_grid=True)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=7.0)
    _add_panel_label(ax2, "b")

    # Panel (c): Cross-Patient Remaining Physical Dose Ceiling + K=2 Multi-Course Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    col_labels = [*pat_labels]
    sc_vals = [p.remaining_physical_p95_gy.get("SpinalCord", 0.0) for p in pats]
    bs_vals = [p.remaining_physical_p95_gy.get("Brainstem", 0.0) for p in pats]
    md_vals = [p.remaining_physical_p95_gy.get("Bone_Mandible", 0.0) for p in pats]

    if cohort.k2_multicourse_output is not None:
        col_labels.append("PT_1\n(K=2 Prior)")
        k2_map = {
            s.tg263_name: s.remaining_physical_p95_gy
            for s in cohort.k2_multicourse_output.result.oar_summaries
        }
        sc_vals.append(k2_map.get("SpinalCord", 0.0))
        bs_vals.append(k2_map.get("Brainstem", 0.0))
        md_vals.append(k2_map.get("Bone_Mandible", 0.0))

    xc = np.arange(len(col_labels))
    wc = 0.25
    for oi, (oar, vals) in enumerate(
        [("SpinalCord", sc_vals), ("Brainstem", bs_vals), ("Bone_Mandible", md_vals)]
    ):
        bars = ax3.bar(
            xc + (oi - 1) * wc,
            vals,
            width=wc * 0.92,
            color=OAR_COLORS[oar],
            label=oar,
            edgecolor=AXES_WHITE,
            linewidth=0.5,
        )
        for b, v in zip(bars, vals):
            if v > 0:
                ax3.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.5,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    color=CHARCOAL_INK,
                )

    ax3.set_xticks(xc)
    ax3.set_xticklabels(col_labels, fontsize=7.8, fontweight="bold")
    ax3.set_ylim(0, max(max(sc_vals), max(bs_vals), max(md_vals), 25.0) * 1.28)
    ax3.set_ylabel(r"Remaining Physical Ceiling $p_{95}$ (Gy)", fontweight="bold")
    ax3.set_title("Cohort & $K=2$ Multi-Course Headroom", fontfamily="serif", pad=40)
    _style_editorial_axes(ax3, horizontal_grid=True)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=7.1, columnspacing=0.8)
    _add_panel_label(ax3, "c")

    footer = (
        "Source data: OpenKBP public clinical dataset (pt_1, pt_2, pt_3; Babier et al., Med Phys 2021, "
        "DOI: 10.1002/mp.14845, CC BY 4.0). Research tool — not a medical device."
    )
    _add_editorial_chrome(
        fig,
        "Multi-Patient OpenKBP Cohort Benchmark: Landmark TRE, Envelope Calibration & K-Course Accumulation",
        footer=footer,
        y_title=0.975,
        y_rule=0.918,
        y_footer=0.020,
        x_left=0.04,
        x_right=0.975,
        title_size=11.6,
    )

    png_path = out_dir / "fig6_cohort_landmark_validation.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig6_cohort_landmark_validation.pdf")
    plt.close(fig)
    return png_path


def plot_fig7_sobol_ntcp_analysis(
    sobol_results: list[SobolSensitivityResult],
    ntcp_summaries: list[OARNTCPSummary],
    out_dir: Path,
) -> Path:
    """Figure 7: Saltelli-Sobol Global Sensitivity (S1 vs ST) & LKB Cumulative NTCP 90% CIs."""
    _configure_academic_style()

    fig = plt.figure(figsize=(14.8, 6.0))
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.15, 1.05],
        wspace=0.36,
        left=0.06,
        right=0.97,
        top=0.73,
        bottom=0.16,
    )

    # Panel (a): Grouped First-Order (S1) vs Total-Order (ST) Sobol Indices across OARs
    ax1 = fig.add_subplot(gs[0, 0])
    param_short = {
        "DIR Spatial Registration": "DIR Spatial\nRegistration",
        "Tissue Recovery Credit": "Tissue\nRecovery",
        "Radiobiology alpha/beta": "Radiobiology\n" + r"$\alpha/\beta$",
        "Course-2 High-Dose Regime": "High-Dose\nRegime",
    }
    pnames = sobol_results[0].parameter_names
    x = np.arange(len(pnames))
    n_oars = len(sobol_results)
    group_w = 0.74 / max(n_oars, 1)

    for oi, sr in enumerate(sobol_results):
        col = OAR_COLORS.get(sr.oar_name, MUTED_INK)
        offset = (oi - (n_oars - 1) / 2.0) * group_w
        s1_vals = [sr.first_order_s1[p] for p in pnames]
        st_vals = [sr.total_order_st[p] for p in pnames]
        ax1.bar(
            x + offset,
            st_vals,
            width=group_w * 0.88,
            color=col,
            alpha=0.24,
            edgecolor=col,
            linewidth=1.0,
            hatch="///",
        )
        bars = ax1.bar(
            x + offset,
            s1_vals,
            width=group_w * 0.88,
            color=col,
            alpha=0.92,
            edgecolor=AXES_WHITE,
            linewidth=0.5,
            label=sr.oar_name,
        )
        for b, st_v in zip(bars, st_vals):
            if st_v >= 0.04:
                ax1.text(
                    b.get_x() + b.get_width() / 2,
                    st_v + 0.022,
                    f"{st_v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.9,
                    color=col,
                    fontweight="bold",
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels([param_short.get(p, p) for p in pnames], fontsize=7.9)
    ax1.set_ylim(0, 1.15)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_ylabel(r"Sobol' Variance Fraction ($S_1$ Solid, $S_T$ Hatched)", fontweight="bold")
    ax1.set_title("Saltelli–Jansen Global Sobol' Sensitivity Indices", fontfamily="serif", pad=42)
    _style_editorial_axes(ax1, horizontal_grid=True)
    handles, _ = ax1.get_legend_handles_labels()
    handles.append(
        mpatches.Patch(
            facecolor=MUTED_INK,
            alpha=0.25,
            edgecolor=MUTED_INK,
            hatch="///",
            label=r"Total-Order $S_T$ (w/ interactions)",
        )
    )
    ax1.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=7.1)
    _add_panel_label(ax1, "a")

    # Panel (b): Cumulative LKB / gEUD NTCP Risk Forest Plot with 90% Uncertainty CIs
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(ntcp_summaries))
    ax2.axvline(5.0, color=FOREST, linestyle="--", linewidth=1.25, label="5% Clinical Risk Ref.")
    ax2.axvline(15.0, color=CRIMSON, linestyle=":", linewidth=1.4, label="15% High-Risk Ref.")

    for i, ns in enumerate(ntcp_summaries):
        col = OAR_COLORS.get(ns.tg263_name, CHARCOAL_INK)
        ax2.plot([ns.ntcp_p05_pct, ns.ntcp_p95_pct], [i, i], color=col, linewidth=3.0, alpha=0.88, zorder=4)
        ax2.scatter(
            [ns.ntcp_nominal_pct],
            [i],
            color=col,
            edgecolor=CHARCOAL_INK,
            linewidth=0.7,
            s=58,
            marker="D",
            zorder=6,
        )
        ax2.scatter([ns.ntcp_worst_pct], [i], color=HEADER_BURGUNDY, s=36, marker="^", zorder=5)
        ax2.text(
            max(ns.ntcp_worst_pct, ns.ntcp_p95_pct) + 2.0,
            i,
            f"{ns.ntcp_nominal_pct:.1f}% [{ns.ntcp_p05_pct:.1f}–{ns.ntcp_p95_pct:.1f}%]",
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold",
            color=CHARCOAL_INK,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [f"{ns.tg263_name}\n({ns.endpoint.split(' (')[0]})" for ns in ntcp_summaries],
        fontsize=7.8,
    )
    ax2.set_ylim(len(ntcp_summaries) - 0.45, -0.78)
    max_risk = max((ns.ntcp_worst_pct for ns in ntcp_summaries), default=30.0)
    ax2.set_xlim(-2.0, min(max(max_risk + 32.0, 46.0), 130.0))
    ax2.set_xlabel("Cumulative LKB NTCP Risk Probability (%)", fontweight="bold")
    ax2.set_title("Cumulative LKB / gEUD NTCP with 90% Uncertainty CIs", fontfamily="serif", pad=42)
    _style_editorial_axes(ax2, horizontal_grid=True)

    f_handles = [
        Line2D(
            [0],
            [0],
            color=CHARCOAL_INK,
            marker="D",
            linewidth=2.4,
            markersize=5.2,
            label="Nominal + 90% CI",
        ),
        Line2D(
            [0],
            [0],
            color=HEADER_BURGUNDY,
            marker="^",
            linestyle="none",
            markersize=5.5,
            label="Worst-case",
        ),
        Line2D([0], [0], color=FOREST, linestyle="--", linewidth=1.25, label="5% Ref."),
        Line2D([0], [0], color=CRIMSON, linestyle=":", linewidth=1.4, label="15% Ref."),
    ]
    ax2.legend(
        handles=f_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=6.8,
        columnspacing=0.8,
    )
    _add_panel_label(ax2, "b", loc="upper right")

    _add_editorial_chrome(
        fig,
        "Joint Global Variance Decomposition (Sobol' QMC) & Cumulative LKB Normal Tissue Complication Risk",
        y_title=0.975,
        y_rule=0.918,
        y_footer=0.020,
        x_left=0.04,
        x_right=0.97,
        title_size=11.6,
    )

    png_path = out_dir / "fig7_sobol_ntcp_analysis.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig7_sobol_ntcp_analysis.pdf")
    plt.close(fig)
    return png_path


def plot_fig8_missing_dicom_reconstruction(
    case: ReIrradiationBenchmarkCase,
    legacy_out: LegacyReconstructionOutput,
    hidden_truth_dose: Volume3D,
    cohort_legacy_audits: list[tuple[str, LegacyReconstructionAudit]],
    out_dir: Path,
) -> Path:
    """Figure 8: Missing-DICOM 3D Prior-Dose Envelope Reconstruction & DICOM Recovery Value."""
    _configure_academic_style()
    _dz, dy, dx = case.ct_course2.spacing_mm
    z_idx, _ = _find_best_slice_indices(case)

    body_ax = case.body_mask_course2[z_idx]
    r0, r1, c0, c1 = _body_crop_bounds(body_ax, pad=6)
    ct_sl = case.ct_course2.data[z_idx, r0:r1, c0:c1]
    bsl = body_ax[r0:r1, c0:c1]

    fig = plt.figure(figsize=(15.0, 9.6))
    gs = GridSpec(
        2,
        6,
        figure=fig,
        hspace=0.38,
        wspace=0.42,
        left=0.055,
        right=0.905,
        top=0.885,
        bottom=0.095,
    )

    vmax = max(float(np.percentile(hidden_truth_dose.data[case.body_mask_course2], 99.5)), 70.0)
    top_specs = [
        (0, 2, hidden_truth_dose.data, "Hidden Ground-Truth 3D Dose (OpenKBP)", "a"),
        (2, 4, legacy_out.dose_nominal_3d.data, "Reconstructed Nominal Dose (PDF Only)", "b"),
        (
            4,
            6,
            legacy_out.dose_upper_3d.data,
            r"Conservative Upper Bound $D_{\mathrm{upper}}(\mathbf{x})$",
            "c",
        ),
    ]

    im_ref = None
    for c_start, c_end, vol_arr, title, plabel in top_specs:
        ax = fig.add_subplot(gs[0, c_start:c_end])
        ax.imshow(ct_sl, cmap="gray", vmin=-250, vmax=450, aspect=dy / dx, interpolation="bilinear")
        d_sl = vol_arr[z_idx, r0:r1, c0:c1]
        masked = np.ma.masked_where((~bsl) | (d_sl < 4.0), d_sl)
        im_ref = ax.imshow(
            masked,
            cmap="magma",
            vmin=4.0,
            vmax=vmax,
            alpha=0.76,
            aspect=dy / dx,
            interpolation="bilinear",
        )
        for oar_name in ["SpinalCord", "Brainstem", "Bone_Mandible", "Parotid_L", "Parotid_R"]:
            if oar_name in case.oar_masks_course2:
                msl = case.oar_masks_course2[oar_name][z_idx, r0:r1, c0:c1]
                if np.any(msl):
                    ax.contour(
                        msl.astype(float),
                        levels=[0.5],
                        colors=[OAR_COLORS[oar_name]],
                        linewidths=1.45,
                    )
        ax.set_title(title, fontfamily="serif", fontsize=9.8, pad=8)
        _style_ct_axes(ax)
        _add_panel_label(ax, plabel, dark_bg=True)
        if plabel == "a":
            _add_scalebar(ax, 30.0, dx)

    if im_ref is not None:
        cbar_ax = fig.add_axes([0.922, 0.56, 0.013, 0.31])
        cbar = fig.colorbar(im_ref, cax=cbar_ax)
        _style_colorbar(cbar, "Prior Physical Dose (Gy)", fontsize=8.4)

    # Panel (d): Blinded OAR DVH Envelopes vs. Hidden Ground-Truth OpenKBP DVH
    ax_d = fig.add_subplot(gs[1, 0:3])
    dose_grid = np.linspace(0.0, 80.0, 320)
    plot_oars = [
        o
        for o in ["SpinalCord", "Brainstem", "Bone_Mandible", "Parotid_L"]
        if o in legacy_out.oar_reconstructed_envelopes
    ]
    for oar in plot_oars:
        env = legacy_out.oar_reconstructed_envelopes[oar]
        col = OAR_COLORS.get(oar, CHARCOAL_INK)
        dvh_lo = compute_dvh_curve(env["dose_lower"], dose_grid)
        dvh_nom = compute_dvh_curve(env["dose_nominal"], dose_grid)
        dvh_hi = compute_dvh_curve(env["dose_upper"], dose_grid)
        ax_d.fill_between(dose_grid, dvh_lo, dvh_hi, color=col, alpha=0.18, linewidth=0)
        ax_d.plot(dose_grid, dvh_nom, color=col, linewidth=1.75, label=f"{oar} (PDF Recon.)")
        if "dose_truth" in env:
            dvh_tr = compute_dvh_curve(env["dose_truth"], dose_grid)
            ax_d.plot(dose_grid, dvh_tr, color=col, linewidth=2.1, linestyle="--")

    ax_d.set_xlim(0, 80)
    ax_d.set_ylim(0, 113)
    ax_d.set_yticks([0, 20, 40, 60, 80, 100])
    ax_d.set_xlabel("Prior Physical Dose (Gy)", fontweight="bold")
    ax_d.set_ylabel("Relative OAR Volume (%)", fontweight="bold")
    safety_pct = legacy_out.audit.truth_voxel_upper_safety_rate_pct or 0.0
    ax_d.set_title(
        f"Blinded DVH Envelope vs. Hidden Truth (Upper Bound Safety: {safety_pct:.1f}%)",
        fontfamily="serif",
        pad=10,
    )
    _style_editorial_axes(ax_d, horizontal_grid=True)
    handles, _ = ax_d.get_legend_handles_labels()
    handles.append(
        mpatches.Patch(
            facecolor=MUTED_INK,
            alpha=0.22,
            label=r"Feasible Band $[D_{\mathrm{low}}, D_{\mathrm{up}}]$",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=CHARCOAL_INK,
            linestyle="--",
            linewidth=2.0,
            label="Hidden Ground Truth (OpenKBP)",
        )
    )
    ax_d.legend(handles=handles, loc="lower left", ncol=2, fontsize=7.2)
    _add_panel_label(ax_d, "d")

    # Panel (e): Quantified "Value of Recovering Original DICOM" Across pt_1..pt_3
    ax_e = fig.add_subplot(gs[1, 3:6])
    x_labels: list[str] = []
    leg_vals: list[float] = []
    dcm_vals: list[float] = []
    deltas: list[float] = []

    for pid, aud in cohort_legacy_audits:
        for oar_key in ("SpinalCord", "Brainstem"):
            match = next((s for s in aud.oar_summaries if s.tg263_name == oar_key), None)
            if match is None or match.true_dicom_remaining_physical_ceiling_gy is None:
                continue
            short_o = "Cord" if oar_key == "SpinalCord" else "BStem"
            x_labels.append(f"{pid.upper()}\n{short_o}")
            leg_vals.append(match.legacy_remaining_physical_ceiling_gy)
            dcm_vals.append(match.true_dicom_remaining_physical_ceiling_gy)
            deltas.append(match.dicom_recovery_value_gy or 0.0)

    x = np.arange(len(x_labels))
    w = 0.36
    b1 = ax_e.bar(
        x - w / 2,
        leg_vals,
        width=w * 0.92,
        color=TERRACOTTA,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label="Legacy PDF-Only Safe Ceiling",
    )
    b2 = ax_e.bar(
        x + w / 2,
        dcm_vals,
        width=w * 0.92,
        color=NAVY,
        edgecolor=AXES_WHITE,
        linewidth=0.5,
        label="Full Ground-Truth DICOM Ceiling",
    )

    for bi, (bar1, bar2, lv, dv, dlt) in enumerate(zip(b1, b2, leg_vals, dcm_vals, deltas)):
        ax_e.text(
            bar1.get_x() + bar1.get_width() / 2,
            lv + 0.45,
            f"{lv:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            color="#8C430E",
        )
        ax_e.text(
            bar2.get_x() + bar2.get_width() / 2,
            dv + 0.45,
            f"{dv:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.9,
            color=NAVY,
        )
        top_y = max(lv, dv) + 2.4
        ax_e.text(
            x[bi],
            top_y,
            f"{dlt:+.1f} Gy",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color=FOREST if dlt >= 0 else CRIMSON,
        )

    ax_e.set_xticks(x)
    ax_e.set_xticklabels(x_labels, fontsize=7.6)
    ax_e.set_ylim(0, max(max(dcm_vals, default=25.0), max(leg_vals, default=25.0)) + 8.0)
    ax_e.set_ylabel("Course-2 Remaining Physical Ceiling (Gy, 5-fx LQL)", fontweight="bold")
    ax_e.set_title(
        r"Value of Recovering Original DICOM ($+\Delta$ Gy Unlocked)",
        fontfamily="serif",
        pad=32,
    )
    _style_editorial_axes(ax_e, horizontal_grid=True)
    ax_e.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=7.3)
    _add_panel_label(ax_e, "e")

    footer = (
        "Blinded validation on OpenKBP (pt_1..pt_3; Babier et al., Med Phys 2021, "
        "DOI: 10.1002/mp.14845, CC BY 4.0): prior 3D dose hidden during reconstruction. "
        "Research tool — not a medical device."
    )
    _add_editorial_chrome(
        fig,
        "Missing-DICOM / Legacy-PDF Physics-Constrained 3D Prior-Dose Envelope Reconstruction (Blinded OpenKBP Validation)",
        footer=footer,
        y_title=0.978,
        y_rule=0.938,
        y_footer=0.018,
        x_left=0.04,
        x_right=0.96,
        title_size=11.7,
    )

    png_path = out_dir / "fig8_missing_dicom_reconstruction.png"
    fig.savefig(png_path, dpi=300)
    fig.savefig(out_dir / "fig8_missing_dicom_reconstruction.pdf")
    plt.close(fig)
    return png_path


def generate_all_publication_figures(
    case: ReIrradiationBenchmarkCase,
    report: AdversarialSweepReport,
    out_dir: str | Path,
) -> list[Path]:
    """Generate core 5 publication-grade academic figures (PNG + PDF) into `out_dir`."""
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return [
        plot_fig1_triplanar_biobudget(case, report, dest),
        plot_fig2_uncertainty_band_dvh(case, report, dest),
        plot_fig3_adversarial_tipping_phase(case, report, dest),
        plot_fig4_schedule_modality_matrix(report, dest),
        plot_fig5_tg132_dir_forensics(case, report, dest),
    ]
