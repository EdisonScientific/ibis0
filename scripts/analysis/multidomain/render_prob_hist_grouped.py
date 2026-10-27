"""Predicted-probability distributions, grouped by training regime, with two summary corners.

A 3x3 grid over the same three groups as `render_belief_revision.py`, so the axis is learned once:

    Qwen3      non-thinking   thinking         | distinct values emitted
    Ibis0      non-thinking   thinking         classifier
    Frontier   GPT-5.6-sol    Opus 5           | mean forecast

Each row groups systems that share a training regime, which puts the three qualitative trends on
different axes of the figure rather than making the reader hold seven panels in mind:

- **Training moves the mass toward the base rate.** The middle row sits on the dashed rule at $0.5$;
  the rows above and below sit far left of it.
- **Untrained systems understate a priori, in two unrelated model families.** That is the top and
  bottom rows together, which is why the frontier systems keep their own row rather than merging into
  a single "untrained" row -- the point is that two families independently land in the same place.
- **Verbalized forecasts are coarsely quantized.** The classifier sits beside the two RL policies, so
  the contrast is between adjacent panels rather than across the figure.

**The two corner panels carry what the histograms cannot.** A KDE smooths over discreteness by
construction, so the quantization claim is invisible in the very panels that are supposed to support it;
the distinct-value chart states it directly. The mean-forecast panel compresses the first and third
trends into one view, since reading centres of mass off seven KDEs is exactly the work a summary should
do for the reader. Both are global summaries occupying the row corners, and both are drawn with a plain
frame so they do not read as members of the row they sit in.

Reads the same cells as `render_prob_hist.py`, whose four-panel figure stays as a diagnostic.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import boot_cache  # noqa: E402
import eval_paths  # noqa: E402
from figure_io import FONT, GREY, new_figure, panel_row, save_figure  # noqa: E402
from multidomain_cells import load_cells  # noqa: E402
from render_prob_hist import _wmean_wsd, weighted_probs  # noqa: E402
from roster_palette import ROSTER, plot_color  # noqa: E402

GROUPS = [
    ("Qwen3", ["base_direct", "base_thinking"]),
    ("Ibis0", ["direct_rl", "thinking_rl", "classifier"]),
    ("Frontier", ["gpt56sol", "opus5"]),
]
# Variant-only labels. Correct ONLY in the `grid` layout, where each row's y-label carries the group
# title ("Qwen3" / "Ibis0") and so supplies the family; there this makes the first two rows read as
# mirrored. Everywhere else it is ambiguous -- base_direct and direct_rl both render as
# "non-thinking" -- so use LABEL.
SHORT = {
    "base_direct": "non-thinking",
    "base_thinking": "thinking",
    "direct_rl": "non-thinking",
    "thinking_rl": "thinking",
    "classifier": "classifier",
    "gpt56sol": "GPT-5.6-sol",
    "opus5": "Opus 5",
}
# Full roster labels, for every layout that sets systems side by side with NO group title on screen:
# the row and block panel titles, the summary y-ticks (which list all seven in ONE axes, so they are
# ambiguous even in the grid), and the console table. Identical to the figure legends by
# construction, which is the point -- a reader maps a panel to a legend row without learning a second
# naming. Opus 5 keeps its short form, as in the belief-revision scatter: the vendor is not in doubt
# and the roster's 'Claude Opus 5' would otherwise be the widest label here.
LABEL = {k: e.label for k, e in ROSTER.items()} | {"opus5": "Opus 5"}

NCOL = 3


def mean_ci(p: np.ndarray, w: np.ndarray, n_boot: int, seed: int = 0):
    """Memoised wrapper; the arithmetic is in `_mean_ci_uncached`. See scripts/lib/boot_cache.py.

    These probe bootstraps were the only uncached ones in the pipeline, and they run the highest draw
    count in it, so every re-render recomputed them from scratch. The key covers the resampled data, the
    draw count, the seed, and the source of this file -- so any change that could move the numbers misses
    the cache rather than serving a stale interval.

    Always returns a tuple: `json` round-trips one to a list, and a value whose TYPE depended on whether
    the cache was warm would be a genuinely nasty bug to chase.
    """
    v = boot_cache.cached(
        "prob_hist.mean_ci",
        {
            "data": boot_cache.fingerprint_records({"p": np.asarray(p).tolist(), "w": np.asarray(w).tolist()}),
            "n_boot": n_boot,
            "seed": seed,
        },
        lambda: _mean_ci_uncached(p, w, n_boot, seed),
        impl_files=[__file__, boot_cache.__file__],
    )
    return None if v is None else tuple(v)


def _mean_ci_uncached(p: np.ndarray, w: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Weighted mean forecast with a percentile bootstrap over forecasts."""
    rng = np.random.default_rng(seed)
    m = float(np.sum(w * p) / np.sum(w))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(p), len(p))
        draws[b] = np.sum(w[i] * p[i]) / np.sum(w[i])
    return m, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _distinct_panel(ax, order, distinct):
    """Log-scaled horizontal bars. The claim this figure's KDEs cannot make."""
    ys = np.arange(len(order))[::-1]
    for y, key in zip(ys, order):
        ax.barh(y, distinct[key], height=0.66, color=plot_color(ROSTER[key]), edgecolor=GREY["ink"], linewidth=0.4)
        # Label outside the bar normally, but inside and right-aligned once the bar is long enough that
        # an outside label would run past the axis. Widening the scale instead would flatten the short
        # bars, which are the other half of the comparison.
        inside = distinct[key] > 400
        ax.text(
            distinct[key] * (0.86 if inside else 1.16),
            y,
            f"{distinct[key]:,}",
            va="center",
            ha="right" if inside else "left",
            fontsize=FONT["annot"],
            color="white" if inside else GREY["ink"],
        )
    ax.set_xscale("log")
    ax.set_xlim(20, 4000)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[k] for k in order], fontsize=FONT["annot"])
    ax.tick_params(axis="x", labelsize=FONT["annot"])
    ax.set_xlabel("distinct values", fontsize=FONT["label"])
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _mean_panel(ax, order, means):
    """Mean forecast against the base rate. Compresses the training and prior-bias trends."""
    ys = np.arange(len(order))[::-1]
    for y, key in zip(ys, order):
        m, lo, hi = means[key]
        ax.errorbar(
            m,
            y,
            xerr=[[m - lo], [hi - m]],
            fmt="o",
            ms=3.4,
            color=plot_color(ROSTER[key]),
            ecolor=GREY["ink"],
            elinewidth=0.9,
            capsize=2,
        )
    ax.axvline(0.5, ls="--", lw=1, color=GREY["ref"], zorder=0)
    ax.set_xlim(0, 1)
    # Same tick text and size as the histogram panels: this axis is the same quantity on the same scale,
    # and "0.0/0.5/1.0" beside their "0/0.5/1" reads as a different axis.
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[k] for k in order], fontsize=FONT["annot"])
    ax.tick_params(axis="x", labelsize=FONT["annot"])
    ax.set_xlabel(r"mean $p(\theta)$", fontsize=FONT["label"])
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# One representative per regime, for the main-body single-row version: an untrained Qwen row, both
# post-trained output channels, and a frontier row. The two summary panels cover the SAME four systems
# rather than the full roster -- at six panels across `TEXT_WIDTH_IN` each is about 0.8in wide, and seven
# y-labels do not fit in that. Both trends survive the cut: the classifier's 1,742 distinct values still
# stand against 54/58/84, and the training shift still reads as 0.22 and 0.27 against 0.50.
CURATED = ["base_thinking", "thinking_rl", "classifier", "opus5"]


def render_row(data, distinct, means, out, bw):
    """Single row for the paper body: four representative distributions, then two full-roster summaries.

    Positioned explicitly rather than through `panel_row`, whose uniform gap cannot give the FIRST
    summary panel the wide left gutter its seven y-tick labels need while giving the histograms, which
    carry no y ticks at all, none. The second summary shares those labels exactly, so it draws them once
    and sits a normal gap away; the reclaimed space goes back into panel width.

    All x labels are pinned to one axes-fraction offset. Left to itself matplotlib places each label
    below its own tick labels, and the summary panels' ticks are a different size from the histograms',
    so the labels would sit at three different heights across a single row.
    """
    import matplotlib

    matplotlib.use("Agg")
    from scipy.stats import gaussian_kde

    keys = [k for k in CURATED if k in data]
    order = [k for _, ks in GROUPS for k in ks if k in means]

    # Margins are sized to the ink, not guessed: measured slack on this figure was 0.23in left,
    # 0.14in right, 0.26in bottom and 0.19in top, all of which is panel width or page height given away
    # for nothing. Each is cut to a ~0.04in buffer. The left margin holds only the rotated "density"
    # label, since the histograms carry no y ticks at all.
    W, H = 6.7, 1.66
    # GUTTER holds the mean panel's y-tick labels, so it is sized to the WIDEST of them:
    # "Qwen3 (non-thinking)" is 0.87in at annot size. Full labels cost ~0.06in off each histogram panel,
    # which is the price of the labels being unambiguous.
    LEFT, RIGHT, HGAP, GUTTER = 0.22, 0.06, 0.18, 0.94
    Y, PH = 0.42, 1.02
    XLAB_Y = -0.235  # axes fraction; one value for every panel in the row
    fixed = LEFT + RIGHT + (len(keys) - 1) * HGAP + GUTTER + HGAP
    unit = (W - fixed) / (len(keys) + 2 * 1.25)
    wh, ws = unit, unit * 1.25
    fig = new_figure(H)

    def _ax(x, w):
        return fig.add_axes((x / W, Y / H, w / W, PH / H))

    bins = np.linspace(0, 1, 31)
    xs = np.linspace(0, 1, 400)
    x = LEFT
    for i, key in enumerate(keys):
        ax = _ax(x, wh)
        x += wh + HGAP
        p, w = data[key]
        col = plot_color(ROSTER[key])
        ax.hist(p, bins=bins, weights=w, density=True, color=col, alpha=0.35, edgecolor="white", linewidth=0.3)
        kde = gaussian_kde(p, bw_method=bw, weights=w)
        dens = kde(xs) + kde(-xs) + kde(2 - xs)
        ax.plot(xs, dens, color=col, lw=1.3)
        ax.fill_between(xs, dens, color=col, alpha=0.18)
        ax.axvline(0.5, ls="--", lw=0.9, color=GREY["ref"], zorder=0)
        ax.set_title(LABEL[key], fontsize=FONT["annot"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        # "0"/"0.5"/"1" rather than matplotlib's "0.0"/"1.0". Every panel keeps all three, so the gap
        # has to hold two overhanging labels: a panel's "1" and its neighbour's "0" each extend past
        # their axis by roughly half a character. At 7pt in a 0.14in gap that collided as "1.00.0";
        # at annot size in a 0.18in gap the two clear each other.
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([])
        ax.tick_params(labelsize=FONT["annot"])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if i == 0:
            ax.set_ylabel("density", fontsize=FONT["label"])

    x += GUTTER - HGAP
    ax_mean = _ax(x, ws)
    _mean_panel(ax_mean, order, means)
    x += ws + HGAP
    ax_dist = _ax(x, ws)
    _distinct_panel(ax_dist, order, distinct)
    # Identical categories, so the labels are drawn once. Ticks stay for the gridline positions.
    ax_dist.set_yticklabels([])
    ax_mean.tick_params(axis="y", labelsize=FONT["annot"])
    shown = {LABEL[k] for k in keys}
    for lbl in ax_mean.get_yticklabels():
        # Systems without a distribution panel of their own recede rather than the ones that have one
        # being emphasised. Weight would read as "these rows matter more", which is not the distinction:
        # every row carries the same statistic, and the only difference is whether its shape also
        # appears to the left. GREY["note"] against GREY["ink"] proved too close to resolve at this size,
        # so the recessive labels take the lighter reference grey.
        lbl.set_color(GREY["ink"] if lbl.get_text() in shown else GREY["ref"])

    for ax in fig.axes:
        ax.xaxis.set_label_coords(0.5, XLAB_Y)
    # The four histograms share one x quantity, so it is named once, centred on their span and pinned to
    # the same height as the per-panel labels on the two summaries.
    hist_span = (LEFT, LEFT + len(keys) * wh + (len(keys) - 1) * HGAP)
    # `va="top"`, matching the anchor matplotlib gives a bottom x-axis label: with `va="center"` this
    # text is centred on the same y the others hang from, so it sits half a line high.
    fig.text(
        sum(hist_span) / 2 / W,
        (Y + XLAB_Y * PH) / H,
        r"$p(\theta)$",
        ha="center",
        va="top",
        fontsize=FONT["label"],
    )

    save_figure(fig, out, tag="prob-hist-row", tight=False)


def render_block(data, distinct, means, out, bw):
    """Two rows x six columns, with spanning cells marking exemption from the row distinction.

        col        0                1                2-3            4            5
        row 0   Qwen3 non-thk    Ibis0 non-thk  [ classifier ] [ frontier ] [ distinct ]
        row 1   Qwen3 thinking   Ibis0 thinking [    2x2     ] [   2x1    ] [   2x1    ]

    Reading DOWN a Qwen3-family column is non-thinking against thinking with training held fixed;
    reading ACROSS from column 0 to column 1 is untrained against post-trained with the output channel
    held fixed. A cell that spans both rows is not part of that dichotomy, which is why the classifier
    and the two frontier models occupy spans: neither has a non-thinking/thinking pair, and stacking
    them in single cells invites a reader who has learned "top row = non-thinking" from the left of the
    figure to misread the right of it.

    The frontier models share one cell because they land in the same place (mean $0.263$ and $0.268$),
    so the overlay states the agreement rather than asking the reader to compare two panels.

    Only ONE summary panel, and it is the distinct-value count. A KDE smooths over discreteness by
    construction, so quantization is invisible in the very panels meant to support it. Mean forecast is
    not summarized separately: it is in every panel title and visible against the dashed rule at $0.5$.
    Dropping it is what frees the full column height for seven legible labels.
    """
    import matplotlib

    matplotlib.use("Agg")
    from scipy.stats import gaussian_kde

    order = [k for _, ks in GROUPS for k in ks if k in distinct]
    W = 6.7
    LEFT, RIGHT, HGAP, GUTTER = 0.40, 0.16, 0.13, 0.94  # GUTTER: see render_row
    PH, RGAP, BOT, TOP = 0.95, 0.44, 0.48, 0.24
    H = BOT + 2 * PH + RGAP + TOP
    SPAN = 2 * PH + RGAP
    wq, wf, ws = 0.80, 0.80, 1.10
    wc = W - LEFT - RIGHT - GUTTER - 3 * HGAP - 2 * wq - wf - ws
    fig = new_figure(H)
    y_top, y_bot = BOT + PH + RGAP, BOT

    def _ax(x, y, wd, ht):
        return fig.add_axes((x / W, y / H, wd / W, ht / H))

    bins = np.linspace(0, 1, 31)
    xs = np.linspace(0, 1, 400)

    def _dens(key):
        p, wt = data[key]
        kde = gaussian_kde(p, bw_method=bw, weights=wt)
        return kde(xs) + kde(-xs) + kde(2 - xs)

    def _frame(ax, show_x, title):
        ax.axvline(0.5, ls="--", lw=0.9, color=GREY["ref"], zorder=0)
        ax.set_title(title, fontsize=FONT["annot"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([])
        ax.tick_params(labelsize=FONT["tick"])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        if show_x:
            ax.set_xlabel(r"$p(\theta)$", fontsize=FONT["annot"])
        else:
            ax.set_xticklabels([])

    def _hist(ax, key, show_x):
        p, wt = data[key]
        col = plot_color(ROSTER[key])
        ax.hist(p, bins=bins, weights=wt, density=True, color=col, alpha=0.35, edgecolor="white", linewidth=0.3)
        dens = _dens(key)
        ax.plot(xs, dens, color=col, lw=1.3)
        ax.fill_between(xs, dens, color=col, alpha=0.18)
        m, _ = _wmean_wsd(p, wt)
        _frame(ax, show_x, f"{LABEL[key]}  (mean {m:.2f})")

    x = LEFT
    for top_key, bot_key in (("base_direct", "base_thinking"), ("direct_rl", "thinking_rl")):
        for key, y, show_x in ((top_key, y_top, False), (bot_key, y_bot, True)):
            if key in data:
                ax = _ax(x, y, wq, PH)
                _hist(ax, key, show_x)
                if abs(x - LEFT) < 1e-9:
                    ax.set_ylabel("density", fontsize=FONT["label"])
        x += wq + HGAP

    if "classifier" in data:
        _hist(_ax(x, y_bot, wc, SPAN), "classifier", True)
    x += wc + HGAP

    front = [k for k in ("gpt56sol", "opus5") if k in data]
    if front:
        ax = _ax(x, y_bot, wf, SPAN)
        for key in front:
            col = plot_color(ROSTER[key])
            ax.plot(xs, _dens(key), color=col, lw=1.4, label=LABEL[key])
            ax.fill_between(xs, _dens(key), color=col, alpha=0.15)
        _frame(ax, True, "frontier")
        ax.legend(loc="upper right", frameon=False, fontsize=FONT["annot"], handlelength=1.0, borderpad=0.1)
    x += wf + GUTTER

    _distinct_panel(_ax(x, y_bot, ws, SPAN), order, distinct)
    ax = fig.axes[-1]
    ax.tick_params(axis="y", labelsize=FONT["annot"])
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in {LABEL[k] for k in data}:
            lbl.set_fontweight("bold")

    save_figure(fig, out, tag="prob-hist-block", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/sensitivity/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--tag", default="ctx*")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out-dir", default="outputs/prob_hist_grouped", help="directory for the figures")
    ap.add_argument("--out-name", default="prob_hist_grouped", help="basename; the layout is suffixed")
    ap.add_argument("--bw", type=float, default=0.2)
    ap.add_argument("--boot", type=int, required=True)
    ap.add_argument(
        "--layout",
        choices=["grid", "row", "block", "all"],
        default="row",
        help="`row` is the published layout and the default; `grid` and `block` are exploratory "
        "variants kept because they answer different questions about the same cells",
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from scipy.stats import gaussian_kde

    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    data, distinct, means = {}, {}, {}
    for _, keys in GROUPS:
        for key in keys:
            cells = load_cells(slug, ROSTER[key].eval_key, root=root, tag=args.tag)
            if not cells:
                print(f"  [skip] {ROSTER[key].label}: no cells")
                continue
            p, w = weighted_probs(cells)
            data[key] = (p, w)
            # Counted on the raw forecasts, unweighted: this is a property of the output channel, not
            # of how domains are pooled.
            distinct[key] = int(
                len(np.unique(np.concatenate([[r["parsed_prob"] for r in rows] for rows in cells.values()])))
            )
            means[key] = mean_ci(p, w, args.boot)

    order = [k for _, ks in GROUPS for k in ks if k in data]
    if not order:
        raise SystemExit("no cells found; nothing to draw")

    base = Path(args.out_dir) / args.out_name
    base.parent.mkdir(parents=True, exist_ok=True)

    L_, R_, GAP_, RGAP = 0.62, 0.12, 0.34, 0.52
    PH, TOP, BOT = 1.18, 0.24, 0.52
    figh = TOP + 3 * PH + 2 * RGAP + BOT
    fig = new_figure(figh)
    rows = [panel_row(fig, NCOL, y=BOT + (2 - r) * (PH + RGAP), h=PH, left=L_, right=R_, gap=GAP_) for r in range(3)]

    bins = np.linspace(0, 1, 31)
    xs = np.linspace(0, 1, 400)
    for r, (title, keys) in enumerate(GROUPS):
        keys = [k for k in keys if k in data]
        for c, key in enumerate(keys):
            ax = rows[r][c]
            p, w = data[key]
            col = plot_color(ROSTER[key])
            ax.hist(p, bins=bins, weights=w, density=True, color=col, alpha=0.35, edgecolor="white", linewidth=0.4)
            kde = gaussian_kde(p, bw_method=args.bw, weights=w)
            dens = kde(xs) + kde(-xs) + kde(2 - xs)
            ax.plot(xs, dens, color=col, lw=1.6)
            ax.fill_between(xs, dens, color=col, alpha=0.18)
            ax.axvline(0.5, ls="--", lw=1, color=GREY["ref"], zorder=0)
            m, sd = _wmean_wsd(p, w)
            ax.set_title(f"{SHORT[key]}  (mean {m:.2f}, sd {sd:.2f})", fontsize=FONT["annot"])
            ax.set_xlim(0, 1)
            ax.tick_params(labelsize=FONT["tick"])
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel(f"{title}\ndensity", fontsize=FONT["label"])
            ax.set_xlabel(r"Model $p(\theta)$", fontsize=FONT["label"])
        # corners
        if r == 0:
            _distinct_panel(rows[0][NCOL - 1], order, distinct)
        elif r == 2:
            _mean_panel(rows[2][NCOL - 1], order, means)
        for c in range(len(keys), NCOL):
            if not (r in (0, 2) and c == NCOL - 1):
                rows[r][c].set_visible(False)

    if args.layout in ("grid", "all"):
        save_figure(fig, base, tag="prob-hist-grouped", tight=False)
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)
    if args.layout in ("row", "all"):
        render_row(data, distinct, means, f"{base}_row", args.bw)
    if args.layout in ("block", "all"):
        render_block(data, distinct, means, f"{base}_block", args.bw)
    print(f"\n{'system':16s} {'mean':>20s}  {'distinct':>9s}")
    for k in order:
        m, lo, hi = means[k]
        print(f"  {LABEL[k]:20s} {m:.3f} [{lo:.3f}, {hi:.3f}]  {distinct[k]:>9,}")


if __name__ == "__main__":
    main()
