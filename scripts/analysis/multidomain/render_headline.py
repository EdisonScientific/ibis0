"""The headline results figure: scalar metrics beside the curves that explain them.

Combines the three separate artifacts (`grouped_metrics`, `reliability`, `roc`) into one figure organised
by *what each number means* rather than by which script happened to produce it:

                  ┌──────────┬─────────┬──────────────┐
    CALIBRATION   │          │ sMECE ↓ │ reliability  │  do the probabilities match frequency?
                  │ Brier ↓  ├─────────┼──────────────┤
    DISCRIMINATION│ (spans)  │ AUROC ↑ │     ROC      │  do they rank cases correctly?
                  └──────────┴─────────┴──────────────┘
                     ──────── shared legend ────────

Each scalar sits next to the curve behind it: the bar is the number, the curve is the shape that number
summarises. Two rows with the same Brier can be over-confident at the top of the range or uniformly
shifted, and only the reliability curve tells them apart; a row can be badly calibrated and still rank
perfectly, which only the ROC panel shows. Putting them adjacent is what makes that reading available
without flipping between figures.

Brier gets its own double-height column because it belongs to BOTH rows rather than to either -- see the
comment on `SPAN_METRIC` for the Murphy-decomposition argument.

**Page geometry.** The paper is single-column with a 6.7 in text width and a 9.25 in text height, so a
macro panel spanning three per-domain columns at a locked aspect would be taller than the page.
Everything here is authored at exactly 6.7 in wide so it drops in at
`width=\\textwidth` with no LaTeX downscaling and the 8-9 pt fonts stay 8-9 pt.

**The two diagonals mean opposite things, so they are drawn differently.** On the reliability panel the
identity line is the *target* -- dashed, the conventional "aspiration" idiom, labelled `perfect
calibration`. On the ROC panel it is the *floor*: drawn as the solid boundary of a filled sub-chance
wedge, so it reads as the edge of a region you want to be outside rather than a line you want to be on.
A reader who sees only the geometry ("2D plot, curves, diagonal") would otherwise carry the reliability
reading straight into the ROC panel and conclude the models are badly calibrated when they rank well.

**One legend, once, at the bottom.** All three source figures overlay the same 7 roster rows in the same
order with the same colours, so they carried three copies of one legend. It spans the full width beneath
both rows rather than sitting inside a panel cell: the rows share one set of models, so a legend parked in
the discrimination row reads as belonging only to that row. Handles are thick line segments, which read
both as a bar swatch and as a curve; a filled `Patch` does not read as a line and a thin `Line2D` does not
read as a bar.

**The reference rule means "what you get for free" in every bar panel.** The constant-`p`-at-base-rate
forecaster lands at Brier 0.25 (the `uncertainty` term), sMECE 0 (a constant forecast is trivially
calibrated) and AUROC 0.5 -- so on the discrimination row the dotted rule agrees with the chance diagonal
in the ROC panel beside it.

**The paper's main text carries the compact variant** (`render_headline_compact.py`: Brier |
reliability with sMECE inset | ROC with AUROC inset, 3.1 in tall). This two-row figure stays as the
full-size diagnostic and shares every drawing primitive and the data loader with it.

Per-domain panels go to `per_domain.{pdf,png}` for the appendix. Coupling them to the macro panel is what
forced the portrait aspect in the first place, and the macro average is the claim the main text makes.

Everything is computed by the modules that own it -- `macro_metrics.build_rows` /
`render_grouped_metrics.draw_metric_panel` for the bars, `render_reliability.macro_curve` for the calibration curve,
`render_roc.curves_for` for the ROC -- so no number here can drift from the same number in the table.
Note AUROC is deliberately NOT appended to the legend labels the way `render_roc` does it: the AUROC bar
panel reports the same statistic *with a CI*, and printing it twice in one figure invites the reader to
wonder which one to trust.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import display_names as dn  # noqa: E402
import eval_paths  # noqa: E402
import render_reliability as rel  # noqa: E402
import render_roc as roc  # noqa: E402
from figure_io import GREY, TEXT_WIDTH_IN, save_figure  # noqa: E402
from macro_metrics import build_rows, constant_reference  # noqa: E402
from multidomain_cells import DOMAINS, load_cells  # noqa: E402
from render_calibration_report import N_BINS, reliability_curve_ci  # noqa: E402
from render_grouped_metrics import draw_metric_panel  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

# Layout, in INCHES, laid out with explicit `fig.add_axes` rather than a gridspec. The panels that must be
# square are square by construction instead of by an aspect lock, which is what makes the geometry
# predictable: `tight_layout` reflows unpredictably around an aspect-locked axes (see the comments in
# render_reliability), and `adjustable="box"` silently letterboxes a panel whose cell is the wrong shape.
FIGW = TEXT_WIDTH_IN  # == the paper's \textwidth
_L, _R = 0.50, 0.22  # left: bar-panel tick labels + ylabel; right: trim + the curve's "1.0" tick
_BW_SPAN = 1.95  # the Brier column, drawn once at double height
_GAP_SPAN = 0.55  # after the Brier column (next panel's ylabel + ticks)
_TOP = 0.50  # column/block headers + panel titles
_ROW_GAP = 0.86  # row-1 xlabel, then row-2's block header
_BOT = 0.44  # row-2 xlabel
_LEG = 0.52  # figure-wide legend strip, two entries per column
# BOTH panels in a row are the same square. If the scalar column took the residual width, every inch
# taken off the curve panel would go to the bars and shift the figure's emphasis to the scalars for a
# reason that has nothing to do with the results. Fixing both at `_SQ` keeps the two halves of each row
# weighted equally at any size.
#
# `_GAP_CURVE` is the residual instead, so leftover width becomes separation between the two squares
# rather than panel area. It still has to hold the curve panel's ylabel and tick labels, so it is never
# purely whitespace.
_SQ = 1.30
_CURVE_W = _SQ
_ROW_H = _SQ
_BW = _SQ  # scalar (sMECE / AUROC) column, square to match the curve beside it
_GAP_CURVE = FIGW - _L - _R - _BW_SPAN - _GAP_SPAN - _BW - _CURVE_W
FIGH = _TOP + _ROW_H + _ROW_GAP + _ROW_H + _BOT + _LEG
_SPAN_H = 2 * _ROW_H + _ROW_GAP  # the Brier panel's height: both rows plus the gap between them

REF_GREY = GREY["ref"]
WEDGE_GREY = GREY["fill"]

# The composite score, drawn ONCE at double height in the left column so it spans both rows.
#
# This is the figure's central claim about what Brier is. The Murphy decomposition gives
# `brier = reliability - resolution + uncertainty`, where `reliability` is a calibration term,
# `resolution` a discrimination term, and `uncertainty = base_rate*(1-base_rate)` a property of the test
# set rather than of any model. On a FIXED test set the third term is a constant, so Brier is a monotone
# combination of exactly the two things the two rows separate -- it belongs to both rows, not to either,
# and a panel spanning both says so geometrically. Filing it under "calibration" would read
# as a claim that Brier is a calibration metric, which it is not: the classifier beats direct RL on Brier
# while being WORSE on sMECE, purely on resolution.
SPAN_METRIC = "brier"
SPAN_HEADER = "Overall"
SPAN_SUB = "proper score"

# (scalar, curve) per block: the row's own scalar next to the curve that shows the shape behind it.
#
# One scalar per row, since Brier has its own column; AUROC is the discrimination number this literature
# quotes. Both terms of the decomposition remain in `master_table`.
#
# Deliberately no second discrimination *curve* either: on a ~50:50 balanced set the obvious candidates
# (gain/lift, precision-at-top-k) are reparametrisations of ROC -- fraction screened is 0.5*(TPR+FPR)
# when the classes are balanced -- so they would occupy a panel to restate the neighbouring one. The
# predicted-probability distributions, the other natural companion, already have render_prob_hist.py.
BLOCKS = (
    ("Calibration", "do probabilities match observed frequency?", "smece", "reliability"),
    ("Discrimination", "do they rank supported above unsupported?", "auroc", "roc"),
)


def _rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Inches -> figure fractions for `fig.add_axes`."""
    return (x / FIGW, y / FIGH, w / FIGW, h / FIGH)


def _jsonable(obj):
    """NaN/Inf -> None, recursively. `json.dumps` emits a bare `NaN`, which is not valid JSON: every
    strict parser rejects the file, so a sidecar meant to be the way results get verified without opening
    a PNG would be unreadable by anything but Python. `null` says "not computed", which is what it means
    (an uncomputed CI when --curve-boot is 0, or an AUROC undefined on a single-class domain)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def _ref_label(ax, x: float, y: float, text: str, va: str) -> None:
    """A reference line's in-panel label, drawn ABOVE the curves with a scrim behind it.

    Kept at rotation 45 and parallel to the line it names, but offset perpendicular to it into whichever
    half-plane the curves do NOT occupy -- see the call sites for which side that is per panel, and why.

    zorder 6 (curves are 3) plus a translucent white box is insurance rather than the primary remedy: it
    keeps a future curve that strays into the empty half-plane from rendering the label unreadable rather than merely
    crowded.
    """
    ax.text(
        x,
        y,
        text,
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va=va,
        fontsize=6.0,
        color=GREY["note"],
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.72),
    )


def _panel_note(ax, text: str) -> None:
    """Method detail in the panel's bottom-right — dead space on both a reliability and an ROC panel
    (curves run bottom-left to top-right on one and hug the top-left on the other), so it costs no
    layout and keeps it out of the title, which has no room."""
    ax.text(0.975, 0.025, text, ha="right", va="bottom", fontsize=6.0, color=GREY["note"], zorder=4)


def _style_square(ax, ticks: tuple[float, ...] = (0, 0.25, 0.5, 0.75, 1.0)) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7.5)
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_reliability(
    ax, curves: dict[str, list[tuple]], *, show_ci: bool, label_ref: bool = True, ticks=(0, 0.25, 0.5, 0.75, 1.0)
) -> None:
    """Reliability curves + the dashed identity TARGET line.

    Dashed, and labelled in-panel rather than in the legend: the reference is a property of this panel's
    geometry, not a series, so a legend entry for it would sit next to the model rows as if it were one.
    """
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=REF_GREY, zorder=1)
    if label_ref:
        # Offset into the LOWER-right triangle, not sat on the diagonal. Every system here is
        # underconfident, so every reliability curve lies above the identity line and the lower triangle
        # is empty -- verified across all models and panels in reliability_curves.json, which holds zero
        # points below it. On the line the label would overprint the very curves the panel is about.
        _ref_label(ax, 0.68, 0.40, "perfect calibration", "bottom")
    for key, pts in curves.items():
        if not pts:
            continue
        col = plot_color(ROSTER[key])
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        ax.plot(xs, ys, "-", color=col, lw=1.5, marker="o", ms=2.6, zorder=3, alpha=0.95)
        if show_ci:
            ax.fill_between(xs, [q[2] for q in pts], [q[3] for q in pts], color=col, alpha=0.13, lw=0, zorder=2)
    _style_square(ax, ticks)


def draw_roc(
    ax,
    grid: np.ndarray,
    curves: dict[str, dict],
    *,
    show_ci: bool,
    label_ref: bool = True,
    ticks=(0, 0.25, 0.5, 0.75, 1.0),
    wedge: bool = True,
) -> None:
    """ROC curves + the sub-chance wedge, whose boundary is the chance FLOOR.

    Filled region rather than a second dashed line: shading says "you want to be outside this", where a
    dashed diagonal identical to the reliability panel's would say "you want to be on this". The constant-
    `p` reference that appears as a rule in the bar panels needs no counterpart here -- a constant forecast
    ranks at chance, so it *is* this diagonal.
    """
    if wedge:
        ax.fill_between([0, 1], [0, 0], [0, 1], color=WEDGE_GREY, lw=0, zorder=0)
    ax.plot([0, 1], [0, 1], ls="-", lw=1.0, color=REF_GREY, zorder=1)
    if label_ref:
        _ref_label(ax, 0.68, 0.655, "chance", "top")
    for key, per in curves.items():
        c = per
        if c is None or not np.isfinite(c["tpr"]).any():
            continue
        col = plot_color(ROSTER[key])
        ax.plot(grid, c["tpr"], "-", color=col, lw=1.5, zorder=3, alpha=0.95)
        if show_ci:
            ax.fill_between(grid, c["lo"], c["hi"], color=col, alpha=0.13, lw=0, zorder=2)
    _style_square(ax, ticks)


def _legend_handles(shown: list[dict], const: dict | None):
    """Thick line segments — the one handle style that reads as both a bar swatch and a curve."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=plot_color(r["entry"]), lw=4.5, solid_capstyle="butt", label=r["label"]) for r in shown
    ]
    if const is not None:
        handles.append(Line2D([], [], color=GREY["accent"], ls=":", lw=1.6, label="constant $p$ = base rate"))
    return handles


def _block_header(
    fig, y_in: float, x0_in: float, x1_in: float, title: str, subtitle: str, *, center: bool = False
) -> None:
    from matplotlib.lines import Line2D

    # Inches -> fractions of THIS figure. Not the module-level FIGW/FIGH: those describe the two-row
    # canvas, and the one-row variant is half as tall, so dividing by FIGH there put every header mid-panel.
    FW, FH = fig.get_size_inches()
    # `center` places the title over the block's midpoint (for variants with no subtitle); the rule still
    # spans the whole block either way.
    tx, ha = ((x0_in + x1_in) / 2 / FW, "center") if center else (x0_in / FW, "left")
    left = fig.text(tx, y_in / FH, title.upper(), fontsize=8.5, weight="bold", va="bottom", ha=ha, color=GREY["ink"])
    right = fig.text(
        x1_in / FW, y_in / FH, subtitle, fontsize=7.0, va="bottom", ha="right", color=GREY["note"], style="italic"
    )
    # The two are anchored to opposite ends of the block, so they collide silently when the block
    # narrows. Nothing errors and nothing clips the canvas; the words simply sit on top of each other. Measured, not
    # trusted.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lw = left.get_window_extent(r).width / fig.dpi
    rw = right.get_window_extent(r).width / fig.dpi
    if lw + rw > (x1_in - x0_in) - 0.10:
        print(
            f"[headline] !! block header '{title}' overflows its {x1_in - x0_in:.2f}in block: "
            f"title {lw:.2f}in + subtitle {rw:.2f}in. Shorten the subtitle in BLOCKS.",
            file=sys.stderr,
        )
    yr = (y_in - 0.045) / FH
    fig.add_artist(Line2D([x0_in / FW, x1_in / FW], [yr, yr], color=GREY["rule"], lw=0.7, transform=fig.transFigure))


def figure_io_style() -> str:
    import figure_io

    return figure_io.STYLE


def save(fig, out_base: Path, label: str) -> list[str]:
    """Write PDF + PNG at the authored size, reporting any clipped edge. See scripts/lib/figure_io.py."""
    return save_figure(fig, out_base, tag=f"headline:{label}", tight=False)


def build_headline(
    rows: list[dict],
    const: dict | None,
    rel_curves: dict[str, list[tuple]],
    roc_curves: dict[str, dict],
    grid: np.ndarray,
    *,
    show_ci: bool,
    macro_bins: int,
):
    """The 2-block figure. Row 1 = calibration, row 2 = discrimination, legend in row 2's spare column.

    Returns the Figure rather than saving it, so a test can assert on geometry without monkeypatching
    `plt.close` or parsing the PDF back out.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [r for r in rows if not np.isnan(r["brier"][0])]
    if not shown:
        print("[headline] no row complete enough to plot", file=sys.stderr)
        return None

    fig = plt.figure(figsize=(FIGW, FIGH))
    x_scalar = _L + _BW_SPAN + _GAP_SPAN
    x_curve = x_scalar + _BW + _GAP_CURVE
    # row 0 = calibration (upper), row 1 = discrimination; both sit above the legend strip.
    y_row = [_BOT + _LEG + _ROW_H + _ROW_GAP, _BOT + _LEG]

    # The composite score first: one tall panel in the left column, spanning both rows and the gap.
    ax_span = fig.add_axes(_rect(_L, y_row[1], _BW_SPAN, _SPAN_H))
    draw_metric_panel(ax_span, shown, const, SPAN_METRIC, title_size=9.5)
    _block_header(fig, y_row[0] + _ROW_H + 0.34, _L, _L + _BW_SPAN, SPAN_HEADER, SPAN_SUB)

    for ri, (title, subtitle, metric, kind) in enumerate(BLOCKS):
        y = y_row[ri]
        # The row header starts AFTER the Brier column -- it labels this row's question, and Brier is not
        # answering it. Spanning the full width would put "CALIBRATION" over a panel that is neither row's.
        _block_header(fig, y + _ROW_H + 0.34, x_scalar, FIGW - _R, title, subtitle)
        ax = fig.add_axes(_rect(x_scalar, y, _BW, _ROW_H))
        draw_metric_panel(ax, shown, const, metric, title_size=9.5)
        # Titles stay SHORT. A panel title is centred on a 1.95 in axes, so a 35-character title at 9.5 pt
        # is ~2.3 in wide and overhangs the canvas -- the last column's title clips off the right edge.
        # The method detail ("10 shared bins", "mean per-domain TPR") goes in an in-panel note at the
        # bottom-right, which is dead space on both a reliability and an ROC panel.
        ax = fig.add_axes(_rect(x_curve, y, _CURVE_W, _ROW_H))
        if kind == "reliability":
            draw_reliability(ax, rel_curves, show_ci=show_ci)
            ax.set_title("Reliability (macro)", fontsize=9.5)
            ax.set_xlabel(r"Model $p(\theta)$", fontsize=8)
            ax.set_ylabel("empirical frequency", fontsize=8)
            _panel_note(ax, f"{macro_bins} shared bins")
        else:
            draw_roc(ax, grid, {k: v.get(roc.MACRO_KEY) for k, v in roc_curves.items()}, show_ci=show_ci)
            ax.set_title("ROC (macro)", fontsize=9.5)
            ax.set_xlabel("false-positive rate", fontsize=8)
            ax.set_ylabel("true-positive rate", fontsize=8)
            _panel_note(ax, "mean per-domain TPR")

    # Figure-wide legend at the bottom, spanning both rows. In the spare discrimination-row cell it read
    # as a legend FOR the discrimination row, which is wrong: the rows share one set of models. Two
    # entries per column, which matplotlib gets from column-major fill at ncol = ceil(n/2).
    handles = _legend_handles(shown, const)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=int(np.ceil(len(handles) / 2)),
        frameon=False,
        fontsize=7.5,
        handlelength=1.6,
        handletextpad=0.55,
        columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.012),
    )
    return fig


# --- appendix: per-domain breakdown -------------------------------------------------------------------
_PD_L, _PD_R, _PD_GAP = 0.60, 0.17, 0.30  # _PD_R > the headline's: the right column's "1.00" tick overhangs
_PD_SQ = (FIGW - _PD_L - _PD_R - 2 * _PD_GAP) / 3
_PD_TOP, _PD_ROW_GAP, _PD_BOT, _PD_LEG = 0.52, 0.82, 0.44, 0.56
PD_FIGH = _PD_TOP + _PD_SQ + _PD_ROW_GAP + _PD_SQ + _PD_BOT + _PD_LEG


def build_per_domain(
    shown: list[dict],
    const: dict | None,
    rel_pd: dict[str, dict[str, list[tuple]]],
    roc_pd: dict[str, dict],
    grid: np.ndarray,
    *,
    show_ci: bool,
    n_bins: int,
):
    """Appendix companion: reliability row over ROC row, one column per domain, one shared legend.

    Same reference-line treatment as the headline figure (dashed target vs shaded sub-chance wedge), so a
    reader who has learned the idiom in the main text does not have to relearn it here. The in-panel
    reference labels are dropped at this size -- they would collide with the curves -- and carried by the
    row headers instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    domains = [d for d in DOMAINS if any(d in v for v in rel_pd.values())]
    if not domains:
        print("[headline] no per-domain curves to plot", file=sys.stderr)
        return None

    def rect(x, y, w, h):
        return (x / FIGW, y / PD_FIGH, w / FIGW, h / PD_FIGH)

    fig = plt.figure(figsize=(FIGW, PD_FIGH))
    y_row = [_PD_BOT + _PD_LEG + _PD_SQ + _PD_ROW_GAP, _PD_BOT + _PD_LEG]
    blocks = (
        ("Calibration", f"reliability, {n_bins} equal-mass bins — dashed diagonal is the target", "reliability"),
        ("Discrimination", "ROC — shaded wedge is sub-chance", "roc"),
    )
    for ri, (title, subtitle, kind) in enumerate(blocks):
        y = y_row[ri]
        yh = (y + _PD_SQ + 0.32) / PD_FIGH
        from matplotlib.lines import Line2D

        fig.text(_PD_L / FIGW, yh, title.upper(), fontsize=8.5, weight="bold", va="bottom", color=GREY["ink"])
        fig.text(
            (FIGW - _PD_R) / FIGW,
            yh,
            subtitle,
            fontsize=7.0,
            va="bottom",
            ha="right",
            color=GREY["note"],
            style="italic",
        )
        fig.add_artist(
            Line2D(
                [_PD_L / FIGW, (FIGW - _PD_R) / FIGW],
                [yh - 0.045 / PD_FIGH] * 2,
                color=GREY["rule"],
                lw=0.7,
                transform=fig.transFigure,
            )
        )
        for i, d in enumerate(domains):
            ax = fig.add_axes(rect(_PD_L + i * (_PD_SQ + _PD_GAP), y, _PD_SQ, _PD_SQ))
            if kind == "reliability":
                draw_reliability(ax, {k: v.get(d, []) for k, v in rel_pd.items()}, show_ci=show_ci, label_ref=False)
                ax.set_xlabel(r"Model $p(\theta)$", fontsize=7.5)
                if i == 0:
                    ax.set_ylabel("empirical frequency", fontsize=7.5)
            else:
                draw_roc(ax, grid, {k: v.get(d) for k, v in roc_pd.items()}, show_ci=show_ci, label_ref=False)
                ax.set_xlabel("false-positive rate", fontsize=7.5)
                if i == 0:
                    ax.set_ylabel("true-positive rate", fontsize=7.5)
            ax.set_title(dn.domain(d), fontsize=9.5)
            if i:
                ax.tick_params(labelleft=False)

    fig.legend(
        handles=_legend_handles(shown, None),
        loc="lower center",
        ncol=min(4, len(shown)),
        frameon=False,
        fontsize=7.5,
        handlelength=1.6,
        bbox_to_anchor=(0.5, 0.005),
    )
    return fig


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Arguments shared with render_headline_compact.py: same cells, same draw counts, same curves."""
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None, help="override the eval tree root")
    ap.add_argument("--keys", default=None, help="comma list of roster keys (default: full roster order)")
    # Defaults MUST match render_master_table's. The bar panels come from the same `build_rows`, so a
    # different draw count here does not just cost time -- it is a different `boot_cache` key, so the table
    # and this figure would compute separate intervals and report different CI widths for the same metric.
    # `render_all.sh` passes both renderers one shared value for the same reason.
    ap.add_argument("--n-boot", type=int, required=True, help="draws for the scalar macro CIs")
    ap.add_argument("--smece-boot", type=int, required=True, help="draws for the sMECE CI (0 = point only)")
    ap.add_argument("--n-bins", type=int, default=N_BINS, help="equal-mass bins for the per-domain curves")
    ap.add_argument("--macro-bins", type=int, default=10, help="shared bins for the macro reliability curve")
    ap.add_argument(
        "--curve-boot",
        type=int,
        required=True,
        help="draws for the curve bands. Pass 0 unless --ci: the bands are not drawn without it, and "
        "computing them costs a full bootstrap per model per panel.",
    )
    ap.add_argument("--allow-partial", action="store_true", help="macro-average rows missing a domain")
    ap.add_argument(
        "--ci",
        action="store_true",
        help="shade the 95%% curve bands. Off by default: with the full roster overlaid they occlude each "
        "other. Implies --curve-boot 200 unless you set it higher.",
    )
    ap.add_argument("--seed", type=int, default=0)


class Inputs:
    """Everything a headline-style figure draws from, loaded once. See `load_inputs`."""

    rows: list[dict]
    const: dict | None
    shown: list[dict]
    grid: np.ndarray
    rel_macro: dict[str, list[tuple]]
    rel_pd: dict[str, dict[str, list[tuple]]]
    roc_all: dict[str, dict]


def load_inputs(args: argparse.Namespace, tag: str = "headline") -> Inputs:
    """Rows + reference + curves for the roster, from the modules that own each number."""
    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else list(ORDER)
    if unknown := [k for k in keys if k not in ROSTER]:
        raise SystemExit(f"unknown roster keys: {unknown} (known: {list(ORDER)})")
    curve_boot = max(args.curve_boot, 200) if args.ci else args.curve_boot

    rows, _ = build_rows(
        slug,
        keys,
        recalibrate=False,
        allow_partial=args.allow_partial,
        n_boot=args.n_boot,
        smece_boot=args.smece_boot,
        root=root,
    )
    if not rows:
        raise SystemExit(f"no results found under {eval_paths.result_dir(slug, '', root=root).parent}")
    inp = Inputs()
    inp.rows = rows
    inp.const = constant_reference(slug, keys, args.n_boot, args.smece_boot, root=root)
    inp.shown = [r for r in rows if not np.isnan(r["brier"][0])]
    inp.grid = np.linspace(0.0, 1.0, roc.N_GRID)
    inp.rel_macro, inp.rel_pd, inp.roc_all = {}, {}, {}
    for r in inp.shown:
        cells = load_cells(slug, r["entry"].eval_key, root=root)
        inp.rel_pd[r["key"]] = {
            d: reliability_curve_ci(
                [x["parsed_prob"] for x in rws], [x["label"] for x in rws], n_bins=args.n_bins, n_boot=curve_boot
            )
            for d, rws in cells.items()
        }
        if len(cells) > 1:
            inp.rel_macro[r["key"]] = rel.macro_curve(cells, args.macro_bins, curve_boot)
        inp.roc_all[r["key"]] = roc.curves_for(cells, inp.grid, curve_boot, args.seed)
        print(
            f"[{tag}] {r['label']:24s} brier={r['brier'][0]:.4f} smece={r['smece'][0]:.4f} "
            f"auroc={r['auroc'][0]:.4f}  macro-rel bins={len(inp.rel_macro.get(r['key'], []))}"
        )
    return inp


def write_sidecar(path: Path, args: argparse.Namespace, inp: Inputs, figure_inches: dict[str, list[float]]) -> None:
    """Every plotted number, as JSON. The house way to verify a render is this file, not the PNG."""
    const = inp.const
    path.write_text(
        json.dumps(
            _jsonable(
                {
                    "testset": args.testset,
                    "revision": args.revision,
                    "tag": "primary",
                    "figure_inches": figure_inches,
                    "blocks": [{"block": b[0], "metrics": list(b[2]), "curve": b[3]} for b in BLOCKS],
                    "scalars": {
                        r["key"]: {
                            "label": r["label"],
                            **{m: list(r[m]) for m in ("brier", "smece", "auroc", "resolution")},
                        }
                        for r in inp.shown
                    },
                    "constant_reference": (
                        {m: list(const[m]) for m in ("brier", "smece", "auroc", "resolution")}
                        if const is not None
                        else None
                    ),
                    "reliability_macro": {
                        k: [{"mean_pred": p[0], "emp_freq": p[1], "lo": p[2], "hi": p[3], "count": p[4]} for p in pts]
                        for k, pts in inp.rel_macro.items()
                    },
                    "roc_macro_auroc": {
                        k: v.get(roc.MACRO_KEY, {}).get("auroc", float("nan")) for k, v in inp.roc_all.items()
                    },
                }
            ),
            indent=2,
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out-dir", default="outputs/headline")
    ap.add_argument("--no-per-domain", action="store_true", help="skip the appendix figure")
    args = ap.parse_args()

    inp = load_inputs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_sidecar(out_dir / "headline.json", args, inp, {"headline": [FIGW, FIGH], "per_domain": [FIGW, PD_FIGH]})
    fig = build_headline(
        inp.rows, inp.const, inp.rel_macro, inp.roc_all, inp.grid, show_ci=args.ci, macro_bins=args.macro_bins
    )
    if fig is not None:
        save(fig, out_dir / "headline", "figure")
    if not args.no_per_domain:
        fig = build_per_domain(
            inp.shown, inp.const, inp.rel_pd, inp.roc_all, inp.grid, show_ci=args.ci, n_bins=args.n_bins
        )
        if fig is not None:
            save(fig, out_dir / "per_domain", "per-domain")
    print(f"[headline] data -> {out_dir / 'headline.json'}")


if __name__ == "__main__":
    main()
