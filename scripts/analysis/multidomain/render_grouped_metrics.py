"""The grouped-metrics bar figure for the published multi-domain TEST set.

`grouped_metrics.{pdf,png}` — one panel per metric, one bar per roster row, with clustered-bootstrap CIs
and the constant-`p` reference drawn as a rule rather than a bar (it frames the comparison rather than
competing in it).

Numbers come from `macro_metrics.py`, whose source hash keys the bootstrap cache, so this file can be
edited freely without invalidating anything.

`draw_metric_panel` is imported by `render_headline.py`, which draws the same bars inside the combined
figure — one implementation, so the two artifacts cannot disagree about the whisker convention, the
detached-interval dagger, or the reference rule.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import display_names as dn  # noqa: E402
from figure_io import FONT, GREY, new_figure, panel_row, save_figure  # noqa: E402
from macro_metrics import METRICS, _outside_ci  # noqa: E402
from roster_palette import plot_color  # noqa: E402
from shared_args import _common_args, resolve, rows_and_reference  # noqa: E402

# Brier / sMECE / AUROC answer the headline question — accuracy, calibration, discrimination — without
# needing the Murphy decomposition explained. reliability/resolution stay available but off by default.
DEFAULT_FIGURE_METRICS: tuple[str, ...] = ("brier", "smece", "auroc")


def _bar_whisker(ax, x: float, pt: float, lo: float, hi: float, half: float = 0.09) -> None:
    """CI whisker at `x` drawn at the interval's OWN coordinates:
    a row whose point falls outside its interval keeps a two-sided [lo, hi] bracket + the detached dagger,
    rather than a clipped one-sided error bar."""
    if np.isnan(lo) or np.isnan(hi):
        return
    detached = _outside_ci(pt, lo, hi)
    col = GREY["flag"] if detached else GREY["ink"]
    ax.plot([x, x], [lo, hi], color=col, lw=1.1, zorder=4)
    for yv in (lo, hi):
        ax.plot([x - half, x + half], [yv, yv], color=col, lw=1.1, zorder=4)
    if detached:
        ax.annotate("\u2020", (x, hi), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=col)


def draw_metric_panel(
    ax, shown: list[dict], const: dict | None, metric: str, *, title_size: float = 11.0, paired: dict | None = None
) -> None:
    """One metric's grouped bars with CI whiskers on `ax`. `const` becomes a reference rule, never a bar.

    Factored out of `render_figure` so `render_headline.py` draws the SAME bars from the same row dicts.
    The alternative -- reimplementing bars in the combined figure -- would let the two artifacts disagree
    about the whisker convention, the detached-interval dagger, or the reference rule, which is precisely
    the class of silent inconsistency this module's one-loader/one-metric-source design exists to prevent.

    `paired` maps a row `key` -> a companion row dict (same shape); where present, its value for `metric`
    is drawn as a second, same-colour bar with a hatched fill beside the primary bar (rows absent from
    `paired` keep a single centred bar). Used to overlay a prompt/condition variant on the primary series.
    """
    vals = []
    for i, r in enumerate(shown):
        pt, lo, hi = r[metric]
        col = plot_color(r["entry"])
        companion = paired.get(r["key"]) if paired else None
        if companion is not None:
            bw, dx = 0.36, 0.21
            ax.bar(i - dx, pt, color=col, width=bw, linewidth=0)
            ppt, plo, phi = companion[metric]
            ax.bar(i + dx, ppt, facecolor=col, width=bw, hatch="////", edgecolor="white", linewidth=0.0)
            vals += [v for v in (pt, hi, ppt, phi) if not np.isnan(v)]
            _bar_whisker(ax, i - dx, pt, lo, hi, half=0.06)
            _bar_whisker(ax, i + dx, ppt, plo, phi, half=0.06)
        else:
            ax.bar(i, pt, color=col, width=0.78, linewidth=0)
            vals += [v for v in (pt, hi) if not np.isnan(v)]
            _bar_whisker(ax, i, pt, lo, hi, half=0.09)
    # The reference level counts toward the span: it can sit above every bar (sMECE, where the trained
    # rows are near zero), and a rule drawn outside the view is a reference silently not shown.
    if const is not None and not np.isnan(const[metric][0]):
        vals.append(const[metric][0])
    span = max(vals) if vals else 1.0
    # The floor is EXACTLY zero, so the bar baseline sits on the axes boundary. Two things depend on that:
    # panels in the same row share a visible origin (a panel with sub-zero headroom has its baseline
    # floating above its neighbour's, which reads as misdrawn), and the baseline lines up with the y=0 of
    # the square curve panels alongside, whose limits are (0, 1).
    #
    # A reference rule AT zero (sMECE, resolution) survives this because `clip_on=False` lets its full
    # width render at the boundary instead of half of it being clipped away -- which is what the old
    # sub-zero headroom was working around. The bottom spine is hidden, so nothing hides the rule.
    ax.set_ylim(0.0, 1.09 * span)
    if const is not None and not np.isnan(const[metric][0]):
        ax.axhline(const[metric][0], color=GREY["accent"], ls=":", lw=1.3, zorder=5, clip_on=False)
    ax.set_title(f"{dn.metric(metric)} ({dn.direction_arrow(metric)})", fontsize=title_size)
    ax.set_xticks([])  # model identity comes from the shared legend, not N repeats of the labels
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=FONT["tick"])
    ax.margins(x=0.04)


def render_figure(rows: list[dict], const: dict | None, metrics: tuple[str, ...], out_base: Path) -> None:
    """Grouped bars, one panel per metric. `const` is drawn as a reference rule, never as a bar."""

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    shown = [r for r in rows if not np.isnan(r[metrics[0]][0])]
    if not shown:
        print("[grouped] nothing complete enough to plot", file=sys.stderr)
        return

    # Absolute inches at TEXT_WIDTH_IN, so LaTeX places the figure at `width=\textwidth` without rescaling
    # and its type matches the headline figure's.
    L_, R_, GAP_ = 0.52, 0.10, 0.40
    PH, TOP, BOT, LEG = 1.95, 0.34, 0.12, 0.50
    fig = new_figure(TOP + PH + BOT + LEG)
    axes = panel_row(fig, len(metrics), y=LEG + BOT, h=PH, left=L_, right=R_, gap=GAP_)
    for ax, metric in zip(axes, metrics):
        draw_metric_panel(ax, shown, const, metric, title_size=FONT["title"])

    handles = [Patch(facecolor=plot_color(r["entry"]), label=r["label"]) for r in shown]
    if const is not None:
        handles.append(Line2D([], [], color=GREY["accent"], ls=":", lw=1.3, label=const["label"]))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=int(np.ceil(len(handles) / 2)),
        frameon=False,
        fontsize=FONT["legend"],
        handlelength=1.5,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.010),
    )
    save_figure(fig, out_base, tag="grouped", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(ap)
    ap.add_argument("--out-dir", default="outputs/master_table")
    ap.add_argument(
        "--figure-metrics",
        default=",".join(DEFAULT_FIGURE_METRICS),
        help=f"comma list of panels to draw, from {list(METRICS)} (default: {','.join(DEFAULT_FIGURE_METRICS)})",
    )
    args = ap.parse_args()

    fig_metrics = tuple(m.strip() for m in args.figure_metrics.split(",") if m.strip())
    if bad := [m for m in fig_metrics if m not in METRICS]:
        raise SystemExit(f"unknown --figure-metrics: {bad} (known: {list(METRICS)})")

    slug, root, keys = resolve(args)
    rows, _, const = rows_and_reference(args, slug, root, keys, recalibrate=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_figure(rows, const, fig_metrics, out_dir / "grouped_metrics")


if __name__ == "__main__":
    main()
