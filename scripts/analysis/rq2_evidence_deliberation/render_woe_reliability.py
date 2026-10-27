"""Weight-of-Evidence reliability figures, drawn through the shared house style (scripts/lib/figure_io.py
+ roster_palette.py) so they match the paper's calibration diagrams: authored at TEXT_WIDTH_IN, shared
FONT scale, roster colours, one curve per system. Consumes analyze_woe_reliability.py's two CSVs.

  woe_reliability_headline   -- slope bars next to a small-multiple grid of every system's curve
  woe_reliability_grid       -- the small multiples alone
  woe_reliability_by_domain  -- the three per-domain panels

Each panel plots realized odds movement (Mantel-Haenszel pooled log-OR of the truth, y) against the
system's claimed weight of evidence (x), both in decibans, relative to the near-neutral (w~=0) reference
tier. The dashed diagonal is the ideal exchange rate; a curve below it overclaims. The slope bars show the
through-origin slope of each system's reliability points (`fit_slope`, the paper's beta); the dashed rule
marks the calibrated slope=1. The main-text figure is drawn by render_woe_reliability_v2.py, which imports
the panel helpers from here.

  uv run --no-sync python scripts/analysis/rq2_evidence_deliberation/render_woe_reliability.py \\
      --stem outputs/bf/woe_reliability [--outdir ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_LIB = Path(__file__).resolve().parents[2] / "lib"  # .../scripts/lib
sys.path.insert(0, str(_LIB))
from figure_io import FONT, GREY, TEXT_WIDTH_IN, new_figure, panel_grid, panel_row, save_figure  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

DECIBAN = 10.0 / np.log(10.0)
DOM_LABEL = {"depmap": "DepMap", "gwas": "GWAS", "tahoe": "Tahoe"}


def _clean_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=FONT["tick"], colors=GREY["ink"])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GREY["rule"])


def _present(points):
    have = set(points["system"])
    return [k for k in ORDER if k in have]


def _sys_xlim(points, key, default_lim):
    """Per-system x range: most systems share +-default_lim, but the base model articulates so far
    (+-25 dB) that it needs its own span; there the identity line is no longer 45 degrees."""
    if key != "base_direct":
        return (-default_lim, default_lim)
    x = points[(points.system == key) & (points.scope == "macro")].x_implied.to_numpy() * DECIBAN
    m = float(np.nanmax(np.abs(x))) * 1.1 if len(x) else default_lim
    return (-m, m)


# The two axes are the two readings of a weight of evidence (a change in log-odds): the ARTICULATED WoE
# is purely the model's stated log-odds change; the REALIZED WoE is purely a truth (theta) quantity, the
# change in the true log-odds. A calibrated evidence-weigher has articulated == realized (the diagonal).
XLAB = "articulated weight of evidence (dB)\n" r"$w=\mathrm{logit}\,p_j-\mathrm{logit}\,p_{j-1}$  (model-stated)"
YLAB = (
    "realized weight of evidence (dB)\n"
    r"$\Delta\,\mathrm{logit}\,\hat P(\theta{=}1)$  (empirical, true label $\theta$)"
)

# Publication colours for the frontier rows.
PAPER_COLOR = {"gpt56sol": "#74aa9c", "opus5": "#dd8669"}
PAPER_LABEL = {
    "base_direct": "Qwen3 (non-thinking)",
    "base_thinking": "Qwen3 (thinking)",
    "direct_rl": "Ibis0 (non-thinking)",
    "thinking_rl": "Ibis0 (thinking)",
    "classifier": "Ibis0 (classifier)",
    "gpt56sol": "GPT-5.6-sol",
    "opus5": "Claude Opus 5",
}
PANEL_LABEL = {
    "base_direct": "Qwen3\n(non-thinking)",
    "base_thinking": "Qwen3\n(thinking)",
    "direct_rl": "Ibis0\n(non-thinking)",
    "thinking_rl": "Ibis0\n(thinking)",
    "classifier": "Ibis0\n(classifier)",
    "gpt56sol": "GPT-5.6-sol",
    "opus5": "Claude Opus 5",
}


def syscolor(key):
    return PAPER_COLOR.get(key, plot_color(ROSTER[key]))


def syslabel(key):
    return PAPER_LABEL[key]


def block_header(fig, x0, x1, axes_top, text):
    from matplotlib.lines import Line2D

    w, h = fig.get_size_inches()
    fig.text(
        (x0 + x1) / (2 * w),
        (axes_top + 0.24) / h,
        text,
        ha="center",
        va="bottom",
        fontsize=FONT["title"],
        weight="bold",
        color=GREY["ink"],
    )
    fig.add_artist(
        Line2D(
            [x0 / w, x1 / w],
            [(axes_top + 0.13) / h] * 2,
            transform=fig.transFigure,
            color=GREY["rule"],
            lw=0.7,
        )
    )


def curve_panel(ax, points, scope, present, xlim, ylim, *, ylabel=True, xlab=XLAB, ylab=YLAB):
    """One reliability panel: identity diagonal + an articulated-vs-realized curve per system (decibans).

    x is the articulated WoE (the model's stated log-odds change); y is the realized WoE (the change in
    the true log-odds of theta, estimated by belief-stratified Mantel-Haenszel). Both are signed through
    the origin; the dashed diagonal is the ideal 1:1 exchange rate, and a curve flatter than it overclaims."""
    ax.axhline(0, color=GREY["rule"], lw=0.7, zorder=0)
    ax.axvline(0, color=GREY["rule"], lw=0.7, zorder=0)
    d0, d1 = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
    ax.plot([d0, d1], [d0, d1], ls=(0, (4, 3)), lw=0.8, color=GREY["ref"], zorder=1)
    for key in present:
        col = syscolor(key)
        sub = points[(points.system == key) & (points.scope == scope)].sort_values("tier")
        if sub.empty:
            continue
        x = sub.x_implied.to_numpy() * DECIBAN
        y = sub.y_realized.to_numpy() * DECIBAN
        ylo = sub.y_lo.to_numpy() * DECIBAN
        yhi = sub.y_hi.to_numpy() * DECIBAN
        ax.errorbar(
            x, y, yerr=[y - ylo, yhi - y], color=col, lw=0, elinewidth=0.7, ecolor=col, alpha=0.5, capsize=1.4, zorder=3
        )
        ax.plot(x, y, color=col, ls="-", lw=1.8, marker="o", ms=3.0, mfc=col, mec=col, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _clean_ax(ax)
    if xlab:
        ax.set_xlabel(xlab, fontsize=FONT["label"])
    if ylabel:
        ax.set_ylabel(ylab, fontsize=FONT["label"])
    else:
        ax.tick_params(labelleft=False)


def beta_panel(ax, betas, present, *, labels=True, title="How much claimed evidence is realized"):
    """Horizontal exchange-rate slope bars --- the through-origin slope of each system's five reliability
    points --- sorted best-to-worst, with the linear-fit R^2 in each label and the calibrated slope=1
    rule. Doubles as the figure's legend (each bar is a labelled colour)."""
    b = betas.set_index("system")
    order = sorted(present, key=lambda k: -b.loc[k, "fit_slope"])  # steepest (best) slope on top
    ypos = np.arange(len(order))[::-1]
    vals = [b.loc[k, "fit_slope"] for k in order]
    lo = [b.loc[k, "fit_slope"] - b.loc[k, "fit_slope_lo"] for k in order]
    hi = [b.loc[k, "fit_slope_hi"] - b.loc[k, "fit_slope"] for k in order]
    cols = [syscolor(k) for k in order]
    ax.barh(ypos, vals, xerr=[lo, hi], color=cols, height=0.62, error_kw=dict(lw=0.7, ecolor=GREY["ink"], capsize=1.6))
    ax.axvline(1.0, color=GREY["ref"], lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(0, max(1.08, max(b.loc[k, "fit_slope_hi"] for k in order) * 1.06))
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks(ypos)
    _clean_ax(ax)
    if labels:
        ax.set_yticklabels(
            [f"{syslabel(k)}  ($R^2\\!=\\!{b.loc[k, 'fit_r2']:.2f}$)" for k in order], fontsize=FONT["tick"]
        )
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlabel("exchange-rate slope", fontsize=FONT["label"])
    if title:
        ax.set_title(title, fontsize=FONT["title"], color=GREY["ink"])


def _legend(fig, present, y):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=syscolor(k), ls=ROSTER[k].linestyle, lw=1.9, marker="o", ms=3.4, label=syslabel(k))
        for k in present
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(4, len(present)),
        frameon=False,
        fontsize=FONT["legend"],
        bbox_to_anchor=(0.5, y),
        handlelength=1.8,
        columnspacing=1.1,
    )


def fig_by_domain(points, betas, outdir, lim):
    present = _present(points)
    xlim = ylim = (-lim, lim)
    W = TEXT_WIDTH_IN
    L, R, LEG, T, H, BOT = 0.62, 0.12, 0.66, 0.30, 1.75, 0.55
    figh = BOT + H + T + LEG
    fig = new_figure(figh, W)
    axes = panel_row(fig, 3, y=BOT, h=H, left=L, right=R, gap=0.30)
    for i, (ax, d) in enumerate(zip(axes, ["tahoe", "depmap", "gwas"])):
        curve_panel(
            ax, points, d, present, xlim, ylim, ylabel=(i == 0), xlab="articulated WoE (dB)", ylab="realized WoE (dB)"
        )
        ax.set_title(DOM_LABEL[d], fontsize=FONT["title"], color=GREY["ink"])
    _legend(fig, present, (BOT + H + T + LEG - 0.06) / figh)
    save_figure(fig, Path(outdir) / "woe_reliability_by_domain", tag="woe_reliability_by_domain")


def fig_combined(points, betas, outdir, lim):
    present = _present(points)
    xlim = ylim = (-lim, lim)
    bmac = betas[betas.scope == "macro"]
    W = TEXT_WIDTH_IN
    L, R = 1.35, 0.12
    LEG, T1, H1, RG, T2, H2, BOT = 0.66, 0.30, 2.15, 0.60, 0.24, 1.30, 0.55
    figh = BOT + H2 + T2 + RG + H1 + T1 + LEG
    fig = new_figure(figh, W)
    y1 = BOT + H2 + T2 + RG  # hero row bottom
    y2 = BOT  # breakdown row bottom
    a_cv, a_bt = panel_row(fig, 2, y=y1, h=H1, left=L, right=R, gap=0.95, widths=[H1, W - L - R - 0.95 - H1])
    curve_panel(
        a_cv, points, "macro", present, xlim, ylim, ylabel=True, xlab="articulated WoE (dB)", ylab="realized WoE (dB)"
    )
    a_cv.set_title("Weight-of-evidence reliability (macro)", fontsize=FONT["title"], color=GREY["ink"])
    beta_panel(a_bt, bmac, present, labels=True)

    bdown = panel_row(fig, 3, y=y2, h=H2, left=L, right=R, gap=0.24)
    for i, (ax, d) in enumerate(zip(bdown, ["tahoe", "depmap", "gwas"])):
        curve_panel(ax, points, d, present, xlim, ylim, ylabel=(i == 0), ylab="realized WoE (dB)")
        ax.set_title(DOM_LABEL[d], fontsize=FONT["note"], color=GREY["note"])
        ax.set_xlabel("articulated WoE (dB)", fontsize=FONT["annot"])
        ax.tick_params(labelsize=FONT["annot"])
    _legend(fig, present, (figh - 0.10) / figh)
    save_figure(fig, Path(outdir) / "woe_reliability", tag="woe_reliability")


def slope_bars_v(ax, betas, present, *, r2: bool = True):
    """Vertical exchange-rate slope bars, sorted best-to-worst, calibrated slope=1 rule; R^2 above each bar
    unless `r2=False`."""
    b = betas.set_index("system")
    order = sorted(present, key=lambda k: -b.loc[k, "fit_slope"])
    x = np.arange(len(order))
    vals = np.array([b.loc[k, "fit_slope"] for k in order])
    lo = vals - np.array([b.loc[k, "fit_slope_lo"] for k in order])
    hi = np.array([b.loc[k, "fit_slope_hi"] for k in order]) - vals
    ax.bar(
        x,
        vals,
        yerr=[lo, hi],
        color=[syscolor(k) for k in order],
        width=0.72,
        error_kw=dict(lw=0.7, ecolor=GREY["ink"], capsize=1.6),
    )
    ax.axhline(1.0, color=GREY["ref"], lw=0.9, ls=(0, (4, 3)), zorder=0)
    top = max(1.1, float((vals + hi).max()) + 0.12)
    ax.set_ylim(0, top)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    for xi, k, h in zip(x, order, hi) if r2 else ():
        ax.text(
            xi,
            b.loc[k, "fit_slope"] + h + top * 0.015,
            f"{b.loc[k, 'fit_r2']:.2f}",
            ha="center",
            va="bottom",
            fontsize=4.9,
            color=GREY["note"],
        )
    _clean_ax(ax)
    ax.tick_params(labelsize=FONT["tick"])
    ax.set_ylabel(r"$\beta_{\mathrm{WoE}}$ (realized / articulated)", fontsize=FONT["label"])


def fig_headline(points, betas, outdir, name):
    """Main figure: a vertical exchange-rate slope bar chart (left) next to a small-multiple grid of every
    system's reliability shape (right). The base model's panel spans its full articulated range, so its
    identity line is deliberately no longer 45 degrees."""
    present = _present(points)
    order = sorted(present, key=lambda k: -betas[betas.scope == "macro"].set_index("system").loc[k, "fit_slope"])
    W = TEXT_WIDTH_IN
    Lm, BARW, MID, R = 0.60, 1.95, 0.58, 0.12
    ncol, nrow, PGAP, RG = 4, 2, 0.16, 0.46
    TOP, BOT, CH = 0.46, 1.02, 1.92
    gx0 = Lm + BARW + MID
    ph = (CH - RG) / nrow
    figh = BOT + CH + TOP
    fig = new_figure(figh, W)
    block_header(fig, Lm, Lm + BARW, BOT + CH, "WEIGHT-OF-EVIDENCE SLOPE")
    block_header(fig, gx0, W - R, BOT + CH, "RELIABILITY")
    slope_bars_v(fig.add_axes((Lm / W, BOT / figh, BARW / W, CH / figh)), betas[betas.scope == "macro"], present)
    axes = panel_grid(fig, nrow, ncol, y=BOT, h=ph, row_gap=RG, left=gx0, right=R, gap=PGAP)
    for i, ax in enumerate(axes):
        if i >= len(order):
            ax.axis("off")
            continue
        key = order[i]
        xl = _sys_xlim(points, key, 8.0)
        curve_panel(ax, points, "macro", [key], xl, (-8, 8), ylabel=False, xlab="", ylab="")
        ax.set_xticks([round(xl[0] * 0.85), 0, round(xl[1] * 0.85)])
        ax.set_yticks([-8, 0, 8])
        ax.tick_params(
            labelbottom=(i >= len(order) - ncol or i + ncol >= len(order)), labelleft=(i % ncol == 0), labelsize=5.4
        )
        ax.set_title(
            PANEL_LABEL[key],
            y=0.93,
            pad=0,
            fontsize=4.9,
            linespacing=0.9,
            color=syscolor(key),
        )
    fig.text(
        (gx0 + (W - R - gx0) / 2) / W,
        0.62 / figh,
        "articulated weight of evidence (dB)",
        ha="center",
        va="bottom",
        fontsize=FONT["note"],
        color=GREY["ink"],
    )
    fig.text(
        (gx0 - 0.44) / W,
        (BOT + CH / 2) / figh,
        "realized (dB)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT["note"],
        color=GREY["ink"],
    )
    _legend(fig, present, 0.14)
    save_figure(fig, Path(outdir) / name, tag=name)


def fig_grid(points, betas, outdir, lim):
    """Small multiples: one panel per system (sorted best-to-worst) so each curve's shape is legible on
    its own, rather than overlaid. Shared axes for comparison; the identity diagonal in every panel."""
    present = _present(points)
    b = betas[betas.scope == "macro"].set_index("system")
    order = sorted(present, key=lambda k: -b.loc[k, "fit_slope"])
    ylim = (-lim, lim)
    W = TEXT_WIDTH_IN
    ncol, nrow = 4, 2
    L, R, TOP, RG, H, BOT = 0.56, 0.14, 0.30, 0.66, 1.34, 0.60
    figh = BOT + nrow * H + (nrow - 1) * RG + TOP
    fig = new_figure(figh, W)
    axes = panel_grid(fig, nrow, ncol, y=BOT, h=H, row_gap=RG, left=L, right=R, gap=0.30)
    for i, ax in enumerate(axes):
        if i >= len(order):
            ax.axis("off")
            continue
        key = order[i]
        bottom_row = i >= len(order) - ncol  # last occupied row (+ any panel above an empty cell)
        curve_panel(
            ax,
            points,
            "macro",
            [key],
            _sys_xlim(points, key, lim),
            ylim,
            ylabel=(i % ncol == 0),
            xlab=("articulated WoE (dB)" if bottom_row else ""),
            ylab="realized (dB)",
        )
        ax.set_title(ROSTER[key].label, fontsize=FONT["note"], color=syscolor(key))
        ax.tick_params(labelsize=FONT["annot"])
    save_figure(fig, Path(outdir) / "woe_reliability_grid", tag="woe_reliability_grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="outputs/bf/woe_reliability", help="input CSV stem (_points/_beta)")
    ap.add_argument("--outdir", default="outputs/figures", help="figure output dir")
    ap.add_argument("--axis-lim", type=float, default=30.0, help="symmetric x/y axis limit in decibans")
    args = ap.parse_args()
    points = pd.read_csv(f"{args.stem}_points.csv")
    betas = pd.read_csv(f"{args.stem}_beta.csv")
    fig_headline(points, betas, args.outdir, "woe_reliability_headline")  # slope bars + per-system grid
    fig_grid(points, betas, args.outdir, 8.0)  # standalone small multiples
    fig_by_domain(points, betas, args.outdir, args.axis_lim)


if __name__ == "__main__":
    main()
