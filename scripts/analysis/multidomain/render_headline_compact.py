"""The paper's main-text headline figure: Brier | reliability (+ sMECE inset) | ROC (+ AUROC inset).

          BRIER (↓)            CALIBRATION              DISCRIMINATION
        ┌──────────┐     ┌──────────────────┐       ┌──────────────────┐
        │ 7 bars   │     │ reliability      │       │ ROC              │
        │ + CI     │     │ curves   ┌─────┐ │       │ curves   ┌─────┐ │
        │          │     │          │sMECE│ │       │          │AUROC│ │
        └──────────┘     └──────────└─────┘─┘       └──────────└─────┘─┘
                         ──────── shared legend ────────

One row, 6.7 x 3.1 in. The two-row figure in `render_headline.py` is 4.9 in tall, about half a text block
once scaled to a page; this is the same content at two thirds the height, and every drawing
primitive (bars, curves, reference labels, legend, eyebrow headers) and the data loader are imported from
there so the two figures cannot disagree on a number or a colour.

**Only the curves get full panels; their scalars become insets.** Both dead corners are the LOWER RIGHT:
every reliability curve lies above the identity line (all systems are underconfident), and on the ROC
panel the lower-right is the sub-chance triangle that no curve enters. The insets are opaque white so
nothing shows through the bars. The one departure from the shared bar convention is the AUROC inset's
floor: at zero, seven bars between 0.66 and 0.82 differ by a few percent of a 0.6 in box and read as "all
the same". Flooring it AT CHANCE (0.5, where the constant-p rule already sits) makes bar height = margin
over chance. The truncation is declared by the 0.5 tick label. The Brier panel keeps the zero floor.

**No panel titles; the eyebrow names the panel.** The row's three block headers are the only labelling,
centred over their panels. The ROC panel has no sub-chance wedge: a grey field behind a white inset box
reads as a badly cropped image rather than as a region. The two diagonals are told apart by
dash pattern and by their labels, which sit at the same anchor below the line on both panels so they read
as a pair.

**Style.** Renders in the Edison house style by default (`scripts/lib/edison_style.py`, matching
Fig. 1): Plex Mono type, ink/muted/hairline neutrals, ruled left-aligned eyebrows,
hairline y-grids. `--style classic` (or `IBIS_FIG_STYLE=classic`) gives the pre-house-style look; layout and
data are identical either way.

Draw counts: identical flags to `render_headline.py` / `render_master_table.py`, for the same reason they
are shared there (a different draw count is a different `boot_cache` key and a different CI width).
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(next(p for p in _HERE.parents if p.name == "scripts") / "lib"))

import edison_style as ed  # noqa: E402
import render_headline as hl  # noqa: E402
import render_roc as roc  # noqa: E402
from figure_io import TEXT_WIDTH_IN  # noqa: E402
from render_grouped_metrics import draw_metric_panel  # noqa: E402

FIGW = TEXT_WIDTH_IN

# Layout in INCHES (see render_headline for why absolute inches rather than a gridspec).
_L = 0.45  # Brier tick labels
_BAR_W, _SQ = 1.30, 1.85  # bar panel width; curve panels are _SQ x _SQ
_GAP = 0.55  # before each curve panel: ylabel + tick labels
_TOP, _BOT, _LEG = 0.32, 0.42, 0.52  # eyebrow band (no panel titles) / xlabel / legend strip
_HDR = 0.16  # eyebrow baseline above the panel top; clears the "1.00" tick label
_R = FIGW - _L - _BAR_W - 2 * _GAP - 2 * _SQ
assert _R >= 0.10, f"layout overfull by {-_R:.2f} in"
FIGH = _TOP + _SQ + _BOT + _LEG

EYEBROW = {"brier": "Brier (↓)", "reliability": "Calibration", "roc": "Discrimination"}
AXLABEL_PT = 8.0
TICK_PT = 6.5  # tick labels on the three full-size panels (the shared default is 7-7.5)

# One anchor for both diagonal labels, BELOW the line and identical on both panels. The Edison anchor sits
# further up the line: Plex Mono sets ~20% wider, and at the classic anchor "perfect calibration" ran off
# the bottom of the axes.
DIAG_LABEL = {"classic": (0.30, 0.16), "edison": (0.36, 0.19)}

# Inset box in axes fractions of the host curve panel: x0, y0, w, h. Sized to the lowest curve, not to the
# diagonal: the classifier's reliability curve dips to y~0.5 at x~0.6 and ~0.57 at x~0.66, so the box
# starts at x=0.64 and stops at y=0.38 (+ ~0.06 for its 7 pt title).
INSET = (0.64, 0.06, 0.33, 0.32)
# metric -> (floor, tick positions). Ticks are pinned so the floor itself is labelled: an auto-locator on
# [0.5, 0.9] picks 0.60/0.75/0.90 and leaves the truncated bottom edge unmarked.
INSET_FLOOR = {"auroc": (0.5, (0.5, 0.7, 0.9))}


def _inset_bars(host, shown: list[dict], const: dict | None, metric: str, *, paired: dict | None = None) -> None:
    """The scalar companion as a small bar panel inside its curve panel."""
    from matplotlib.ticker import MaxNLocator

    ax = host.inset_axes(INSET)
    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.set_zorder(7)  # above curves (3) and reference labels (6)
    draw_metric_panel(ax, shown, const, metric, title_size=7.0, paired=paired)
    ax.tick_params(labelsize=5.5, length=2, pad=1.5)
    if metric in INSET_FLOOR:
        floor, ticks = INSET_FLOOR[metric]
        ax.set_ylim(floor, ax.get_ylim()[1])
        ax.set_yticks(list(ticks))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.spines["left"].set_linewidth(0.6)
    return ax


def build(inp: hl.Inputs, *, show_ci: bool, style: str = "classic", paired: dict | None = None):
    """The figure. Returns it rather than saving so a test can assert on geometry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    shown = inp.shown
    if not shown:
        print("[headline-compact] no row complete enough to plot", file=sys.stderr)
        return None
    edison = style == "edison"

    def header(x0, x1, label):
        if edison:
            ed.eyebrow(fig, x0, y_hdr, x1 - x0, label)
        else:
            hl._block_header(fig, y_hdr, x0, x1, label, "", center=True)

    def finish(ax, *, bars: bool, tick_pt: float = TICK_PT):
        if edison:
            ed.style_axes(ax, tick_pt=tick_pt)
            if bars:
                ed.edge_bars(ax)
        else:
            ax.tick_params(labelsize=tick_pt)

    # Axis labels follow Fig. 1's pairing: words in Helvetica(-class) sans beside Computer Modern math.
    # Ticks, eyebrows and legend stay Plex Mono.
    label_font = {"fontfamily": ed.sans()} if edison else {}

    fig = plt.figure(figsize=(FIGW, FIGH))
    y = _BOT + _LEG
    y_hdr = y + _SQ + _HDR

    def rect(x, w):
        return (x / FIGW, y / FIGH, w / FIGW, _SQ / FIGH)

    x = _L
    ax = fig.add_axes(rect(x, _BAR_W))
    draw_metric_panel(ax, shown, inp.const, hl.SPAN_METRIC, title_size=9.5, paired=paired)
    ax.set_title("")  # the eyebrow carries the name
    finish(ax, bars=True)
    header(x, x + _BAR_W, EYEBROW["brier"])
    x += _BAR_W + _GAP

    for _title, _subtitle, metric, kind in hl.BLOCKS:
        ax = fig.add_axes(rect(x, _SQ))
        if kind == "reliability":
            hl.draw_reliability(ax, inp.rel_macro, show_ci=show_ci, label_ref=False)
            hl._ref_label(ax, *DIAG_LABEL[style], "perfect calibration", "bottom")
            # Lowercase p: the paper writes p(theta | I) throughout, and Fig. 1 labels the prior p(theta).
            ax.set_xlabel(r"Model $p(\theta)$", fontsize=AXLABEL_PT, **label_font)
            ax.set_ylabel(r"Empirical $p(\theta)$", fontsize=AXLABEL_PT, **label_font)
        else:
            macro = {k: v.get(roc.MACRO_KEY) for k, v in inp.roc_all.items()}
            hl.draw_roc(ax, inp.grid, macro, show_ci=show_ci, wedge=False, label_ref=False)
            hl._ref_label(ax, *DIAG_LABEL[style], "chance", "bottom")
            ax.set_xlabel("FPR", fontsize=AXLABEL_PT, **label_font)
            ax.set_ylabel("TPR", fontsize=AXLABEL_PT, **label_font)
        finish(ax, bars=False)
        inset = _inset_bars(ax, shown, inp.const, metric, paired=paired)
        if edison:
            ed.style_axes(inset, tick_pt=5.5)
            inset.tick_params(length=2, pad=1.5)
            ed.edge_bars(inset)
        header(x, x + _SQ, EYEBROW[kind])
        x += _SQ + _GAP

    handles = hl._legend_handles(shown, inp.const)
    if paired:
        from matplotlib.patches import Patch

        handles = list(handles) + [
            Patch(facecolor="#8a8f96", edgecolor="white", hatch="////", linewidth=0.0, label="+ 0.5-prior prompt"),
        ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=int(np.ceil(len(handles) / 2)),
        frameon=False,
        fontsize=6.8 if edison else 7.5,  # Plex Mono sets ~20% wider than DejaVu Sans
        handlelength=1.6,
        handletextpad=0.55,
        columnspacing=1.2 if edison else 1.6,
        bbox_to_anchor=(0.5, 0.012),
    )
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    hl.add_common_args(ap)
    # Defaults to the repo-wide switch (figure_io.STYLE, from IBIS_FIG_STYLE) so render_all.sh needs no flag.
    ap.add_argument("--style", choices=["classic", "edison"], default=hl.figure_io_style())
    ap.add_argument("--out-dir", default="outputs/headline_compact")
    ap.add_argument(
        "--p05-eval-root",
        default=None,
        help="second eval tree; rows in --p05-keys get a same-colour hatched companion bar (0.5-prior variant)",
    )
    ap.add_argument("--p05-testset", default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}")
    ap.add_argument("--p05-keys", default=None, help="comma keys to draw a companion bar for (default: none)")
    ap.add_argument(
        "--sort-metric",
        default=None,
        help="reorder rows worst->best by this metric's primary-series point (e.g. brier), across all bar panels",
    )
    args = ap.parse_args()
    if args.style == "edison":
        ed.install()
        hl.REF_GREY = ed.BORDER_STRONG  # module constants captured from GREY at import time
        hl.WEDGE_GREY = ed.CARD

    inp = hl.load_inputs(args, tag="headline-compact")

    if args.sort_metric and inp.shown:
        # worst -> best by the primary (baseline) series; higher Brier/sMECE is worse, so descending point.
        inp.shown.sort(key=lambda r: r[args.sort_metric][0], reverse=True)

    paired = None
    if args.p05_eval_root and args.p05_keys:
        p05_keys = [k.strip() for k in args.p05_keys.split(",") if k.strip()]
        p05_slug = hl.eval_paths.testset_slug(args.p05_testset, args.revision)
        p05_rows, _ = hl.build_rows(
            p05_slug,
            p05_keys,
            recalibrate=False,
            allow_partial=True,
            n_boot=args.n_boot,
            smece_boot=args.smece_boot,
            root=Path(args.p05_eval_root),
        )
        paired = {r["key"]: r for r in p05_rows}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hl.write_sidecar(out_dir / "headline_compact.json", args, inp, {"headline_compact": [FIGW, FIGH]})
    fig = build(inp, show_ci=args.ci, style=args.style, paired=paired)
    if fig is not None:
        hl.save_figure(fig, out_dir / "headline_compact", tag="headline-compact", tight=paired is not None)
    print(f"[headline-compact] data -> {out_dir / 'headline_compact.json'}")


if __name__ == "__main__":
    main()
