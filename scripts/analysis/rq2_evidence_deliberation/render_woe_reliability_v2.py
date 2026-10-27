"""The paper's WoE reliability figure: slope bars + four reliability panels + the Tahoe evidence hierarchy.

Builds on the panel helpers in `render_woe_reliability.py` and makes room for the Tahoe panel of the
appendix staircase (`multidomain/render_woe_structure.fig_staircase`) by showing reliability small
multiples for four systems: Qwen3 (non-thinking), Ibis0 (non-thinking), GPT-5.6-sol, Claude Opus 5. The
slope bars (`fit_slope`, the paper's beta) keep all seven systems -- beta is the figure's summary
statistic and the text quotes it for every system; the panels only show four representative shapes.

One row: 2x2 WoE-vs-outcomes panels | beta bars | Tahoe evidence hierarchy as lollipops in the expert tier
order -> woe_reliability_v2.{pdf,png} (the paper's woe_reliability.pdf). `--all-variants` also writes the
step/dodge/lollipop/bars comparison.

Everything numeric comes from the modules that own it: the reliability points and slopes from
`analyze_woe_reliability.py`'s CSVs (`--stem`), the staircase from `render_woe_structure`'s own loader and
profile functions over the eval tree. The staircase's tier ORDER is derived from all seven systems, exactly
as in the appendix figure, so the Tahoe panel here and there order the tiers identically; only four
systems are drawn.

  python scripts/analysis/rq2_evidence_deliberation/render_woe_reliability_v2.py \\
      --stem ibis_data/woe_reliability/woe_reliability --outdir outputs/woe_reliability_v2
"""

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).absolute().parent  # absolute, NOT resolve: a symlinked copy uses the lib next to it
_SCRIPTS = next(p for p in _HERE.parents if p.name == "scripts")
sys.path.insert(0, str(_SCRIPTS / "lib"))
sys.path.insert(0, str(_SCRIPTS / "analysis" / "multidomain"))
sys.path.insert(0, str(_HERE))

import eval_paths  # noqa: E402

# Imported BEFORE render_woe_reliability, which puts its own resolved scripts/lib first on sys.path: whichever
# figure_io loads first is the one every module shares (sys.modules), and it must be this script's.
import figure_io  # noqa: E402, F401
import render_woe_reliability as wr  # noqa: E402
import render_woe_structure as ws  # noqa: E402
from figure_io import FONT, GREY, TEXT_WIDTH_IN, new_figure, save_figure  # noqa: E402
from roster_palette import ORDER, ROSTER  # noqa: E402

PANEL_KEYS = ("direct_rl", "base_direct", "gpt56sol", "opus5")  # 2x2 row-major: Ibis0, Qwen3 / GPT, Claude
STAIR_DOMAIN = "tahoe"
REL_LIM = 8.0  # decibans; the base model gets its own wider x-range (wr._sys_xlim)
PANEL_TITLE = {k: wr.PAPER_LABEL[k] for k in PANEL_KEYS}


# --- staircase data -------------------------------------------------------------------------------------
def staircase_data(testset: str, revision: str, tag: str, eval_root: str | None):
    """(tier order, {key: {tier: normalized |WoE|}}) for STAIR_DOMAIN, order derived from all 7 systems."""
    slug = eval_paths.testset_slug(testset, revision)
    root = Path(eval_root) if eval_root else None
    meta = ws.stub_meta(testset, revision=revision)
    profiles = {}
    for key in ORDER:
        recs, _ = ws.woe_records(slug, ROSTER[key].eval_key, tag, root, meta)
        if recs:
            profiles[key] = ws.tier_profile(recs, STAIR_DOMAIN)
    order = ws.empirical_order({k: p for k, p in profiles.items() if p}, STAIR_DOMAIN)
    return order, profiles


def _tier_label(name: str) -> str:
    """Break a tier name at its comma ("other drug, same cell" -> two clauses), else wrap at 10 chars.

    A plain character-count wrap split "other drug, same cell" into three ragged lines."""
    return name.replace(", ", ",\n") if ", " in name else textwrap.fill(name, 10)


STAIR_STYLES = ("step", "dodge", "lollipop", "bars")


def staircase_panel(ax, order, profiles, keys, *, style: str = "step", tick_pt: float = 5.2) -> None:
    """One domain's tier profile. Three encodings, same numbers:

    step   the appendix figure's step lines (risers between tiers).
    dodge  no risers: each model is a dot in a fixed slot inside each tier column (a flat tick under each
           dot was tried: at ~0.05 in per slot the marker covers it), on a log2 axis (normalized WoE is a ratio to the episode mean, so 2x and 1/2x sit
           equidistant from 1 and the one large value no longer squashes the cluster near 1).
    lollipop  dodge + a hairline stem from the episode mean to each dot, so the offset from 1 is the mark
           (lighter than bars, which would repeat the WoE calibration panel's bar vocabulary).
    bars   grouped bars rising or falling from the episode mean, same log2 axis.
    """
    from matplotlib.ticker import FixedLocator, NullLocator

    assert style in STAIR_STYLES, style
    ax.axhline(1.0, color=GREY["ref"], ls="--", lw=0.9, zorder=1)
    x = np.arange(len(order))
    n = len(keys)
    slot = 0.72 / n  # the column's usable width, split into one slot per model
    vals = []
    for j, key in enumerate(keys):
        ys = np.array([profiles.get(key, {}).get(t, np.nan) for t in order], dtype=float)
        vals += [v for v in ys if np.isfinite(v)]
        col = wr.syscolor(key)
        if style == "step":
            ax.step(x, ys, where="mid", color=col, lw=1.5, alpha=0.95, zorder=3)
            ax.plot(x, ys, ls="none", marker="o", ms=2.2, color=col, zorder=4)
            continue
        xo = x + (j - (n - 1) / 2) * slot
        if style in ("dodge", "lollipop"):
            if style == "lollipop":
                ax.vlines(xo, 1.0, ys, color=col, lw=1.0, zorder=3)
            ax.plot(xo, ys, ls="none", marker="o", ms=3.4, color=col, zorder=4)
        else:
            ax.bar(xo, ys - 1.0, bottom=1.0, width=slot * 0.86, color=col, zorder=3)
    if style != "step":
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
        ax.yaxis.set_minor_locator(NullLocator())
        # Ratio labels: on a log axis "0.5× / 1× / 2×" says "half / twice as influential as the episode's average
        # stub", and makes the symmetry about the mean explicit.
        ax.yaxis.set_major_formatter(lambda v, _: {0.5: "0.5\u00d7", 1.0: "1\u00d7", 2.0: "2\u00d7"}.get(v, f"{v:g}"))
        # faint column boundaries: the grouping unit is the tier, not the model
        for b in x[:-1] + 0.5:
            ax.axvline(b, color=GREY["rule"], lw=0.5, zorder=0)
    ax.set_xticks(x)
    hier = ws.HIER[STAIR_DOMAIN]
    # Horizontal and wrapped, not rotated: rotated 40 deg the tier names needed a ~0.9 in band under the
    # panel; stacked on two or three short lines they need ~0.35 in. The curated rank "[n]" gets its own line
    # so a reordering against the expert hierarchy still reads at a glance.
    ax.set_xticklabels(
        [f"[{hier.index(t) + 1}]\n" + _tier_label(ws.TIER_SHORT.get(t, t)) for t in order],
        fontsize=tick_pt,
        linespacing=1.0,
    )
    ax.set_xlim(-0.5, len(order) - 0.5)
    # Labelled in place rather than in the legend: this dashed line is the episode mean (1 by construction),
    # a different reference from the dashed "ideal" lines in the other two panels.
    # Two lines so it stays inside the last tier column, which has no upward stems in either ordering; on
    # one line it ran across the neighbouring column's tallest lollipop.
    ax.text(
        len(order) - 0.55,
        1.0,
        "episode\nmean",
        ha="right",
        va="bottom",
        fontsize=5.2,
        linespacing=1.0,
        color=GREY["note"],
    )
    if style == "step":
        ax.set_ylim(min(vals + [1.0]) * 0.9, max(vals) * 1.06)
    else:
        ax.set_ylim(min(vals + [1.0]) * 0.85, max(vals) * 1.12)
    wr._clean_ax(ax)  # resets every tick label to FONT["tick"], so the x size is re-applied after it
    ax.tick_params(axis="y", labelsize=FONT["tick"])
    ax.tick_params(axis="x", labelsize=tick_pt, pad=2.0)
    # No unit: this is |WoE| divided by the episode-mean |WoE|, a ratio of two decibans quantities.
    ax.set_ylabel("Normalized WoE", fontsize=FONT["label"])


# --- reliability panels ---------------------------------------------------------------------------------
def rel_panel(ax, points, key, *, labelleft: bool, labelbottom: bool, tick_pt: float = 5.4) -> None:
    xl = wr._sys_xlim(points, key, REL_LIM)
    wr.curve_panel(ax, points, "macro", [key], xl, (-REL_LIM, REL_LIM), ylabel=False, xlab="", ylab="")
    ax.set_xticks([round(xl[0] * 0.85), 0, round(xl[1] * 0.85)])
    ax.set_yticks([-REL_LIM, 0, REL_LIM])
    ax.tick_params(labelleft=labelleft, labelbottom=labelbottom, labelsize=tick_pt)
    ax.set_title(PANEL_TITLE[key], fontsize=5.4, color=wr.syscolor(key), pad=2.5, **_title_font())


def _title_font() -> dict:
    """Panel titles are words: sans under the Edison style (see scripts/lib/edison_style.py), default otherwise."""
    if getattr(figure_io, "STYLE", "classic") == "edison":
        import edison_style

        return {"fontfamily": edison_style.sans()}
    return {}


def _ordered_panels(betas):
    b = betas.set_index("system")
    return [k for k in PANEL_KEYS if k in b.index]  # fixed order, not by slope


def _legend(fig, present, y_frac, ncol=4):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=wr.syscolor(k), lw=4.0, solid_capstyle="butt", label=wr.syslabel(k)) for k in present
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        fontsize=FONT["legend"],
        bbox_to_anchor=(0.5, y_frac),
        handlelength=1.6,
        columnspacing=1.2,
    )


# --- layouts --------------------------------------------------------------------------------------------
def fig_row(points, betas, stair, out: Path, stair_style: str = "step", name: str | None = None) -> None:
    """2x2 WoE-vs-outcomes | beta bars | Tahoe staircase, one row.

    The panels showing the raw relationship come first and the slope that summarises them second, so the
    beta bars read as "the slope of the curves you just saw".
    """
    order, profiles = stair
    present = wr._present(points)
    panels = _ordered_panels(betas)
    W = TEXT_WIDTH_IN
    L, R = 0.50, 0.10
    SQ, PG, G1, BARW, G2 = 0.80, 0.10, 0.52, 1.35, 0.55  # G1 holds the beta axis label + ticks
    GRIDW = 2 * SQ + PG
    STW = W - L - R - GRIDW - G1 - BARW - G2
    PH = 2 * SQ + 0.26  # 2x2 block height incl. the inner title gap
    TOP, XT, LEG = 0.44, 0.52, 0.36  # headers / wrapped staircase tick labels + x-labels / legend
    figh = LEG + XT + PH + TOP
    y0 = LEG + XT
    fig = new_figure(figh, W)

    def rect(x, y, w, h):
        return (x / W, y / figh, w / W, h / figh)

    x = L
    # not "Reliability": that names the headline figure's probability diagram
    wr.block_header(fig, x, x + GRIDW, y0 + PH, "WoE vs. outcomes")
    for i, key in enumerate(panels):
        r, c = divmod(i, 2)
        ax = fig.add_axes(rect(x + c * (SQ + PG), y0 + (1 - r) * (SQ + 0.26), SQ, SQ))
        rel_panel(ax, points, key, labelleft=(c == 0), labelbottom=(r == 1))
        if i == 0:  # name the dashed diagonal once; it means the same thing as the beta panel's "ideal"
            ax.text(
                REL_LIM * 0.80,
                REL_LIM * 0.58,
                "ideal",
                ha="right",
                va="top",
                rotation=45,
                rotation_mode="anchor",
                fontsize=5.2,
                color=GREY["note"],
            )
    fig.text(
        (x + SQ + PG / 2) / W,
        (y0 - 0.30) / figh,
        r"Weight of evidence $w$ (dB)",
        ha="center",
        va="top",
        fontsize=FONT["label"],
    )
    fig.text(
        (x - 0.30) / W,
        (y0 + PH / 2) / figh,
        r"Odds ratio of support $\psi$ (dB)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT["label"],
    )
    x += GRIDW + G1
    wr.block_header(fig, x, x + BARW, y0 + PH, "WoE calibration")
    ax_b = fig.add_axes(rect(x, y0, BARW, PH))
    wr.slope_bars_v(ax_b, betas, present, r2=False)
    ax_b.text(len(present) - 0.5, 1.0, "ideal", ha="right", va="bottom", fontsize=5.2, color=GREY["note"])
    ax_b.set_ylabel(r"$\beta$", fontsize=FONT["label"])
    x += BARW + G2
    wr.block_header(fig, x, x + STW, y0 + PH, "Evidence hierarchy (Tahoe)")
    staircase_panel(fig.add_axes(rect(x, y0, STW, PH)), order, profiles, panels, style=stair_style)
    _legend(fig, present, 0.01)
    name = name or "woe_reliability_v2_row" + ("" if stair_style == "step" else f"_{stair_style}")
    save_figure(fig, out / name, tag=f"woe_v2:{name}", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default="outputs/bf/woe_reliability", help="analyze_woe_reliability CSV stem")
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/sensitivity/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--tag", default="ctx*", help="dense-sweep cells (the staircase needs k-1 -> k increments)")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--outdir", default="outputs/woe_reliability_v2")
    ap.add_argument("--all-variants", action="store_true", help="also write the step/dodge/lollipop/bars comparison")
    args = ap.parse_args()

    points = pd.read_csv(f"{args.stem}_points.csv")
    betas = pd.read_csv(f"{args.stem}_beta.csv")
    betas = betas[betas.scope == "macro"]
    stair = staircase_data(args.testset, args.revision, args.tag, args.eval_root)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    # The paper figure: lollipops in the CURATED (expert) tier order -> woe_reliability_v2.{pdf,png}. Steps
    # zig-zag in the curated order; unconnected
    # lollipops do not, so a departure from the expert hierarchy reads as an out-of-order height.
    curated = [t for t in ws.HIER[STAIR_DOMAIN] if t in stair[0]]
    fig_row(points, betas, (curated, stair[1]), out, stair_style="lollipop", name="woe_reliability_v2")
    if args.all_variants:  # alternative encodings, in the data-derived order
        for st in STAIR_STYLES:
            fig_row(points, betas, stair, out, stair_style=st)


if __name__ == "__main__":
    main()
