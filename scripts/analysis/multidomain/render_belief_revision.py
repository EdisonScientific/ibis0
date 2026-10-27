"""Both belief-revision probes in one figure, for the main results section.

Two rows, one per probe, over the same three groups of systems:

    row 1   position-1 bias            does the same stub count for more when it arrives first?
    row 2   belief-consistency violation   does a conflicting pair move the belief as the parts predict?

    columns   Qwen3 | Ibis0 | Frontier

The grouping is what makes the figure worth having as a composite. Within each group the systems are
ordered so that a direct/thinking pair sits adjacent, because the finding in row 1 is a within-pair
difference: verbalizing a reasoning trace lowers first-position bias by about the same amount whether or
not the model has been post-trained. Column membership and within-column order are IDENTICAL in both
rows, so a reader can follow one system down the figure; sorting each row by its own metric would break
that and is why this does not reuse the per-probe renderers' orderings.

Rows do not share a y axis -- the two probes are different quantities in different ranges (row 1 spans
about $\\pm1$ dB, row 2 reaches $7.5$) -- but panels within a row do, which is the comparison that
matters.

The per-probe appendix figures (`render_woe_position.py`, `render_conflict_additivity.py`) remain the
detailed presentation; this is the summary. All three read the same cells, so no number can disagree.
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS = next(p for p in _HERE.parents if p.name == "scripts")
sys.path.insert(0, str(_SCRIPTS / "lib"))
sys.path.insert(0, str(_HERE))

import figure_io  # noqa: E402
from multidomain_cells import DOMAINS, load_cells  # noqa: E402
from render_conflict_additivity import episode_residuals, macro_ci  # noqa: E402
from render_woe_position import episode_displacements, first_position_bias  # noqa: E402
from roster_palette import ROSTER, plot_color  # noqa: E402

CONFLICT_DOMAINS = ("tahoe", "depmap")  # GWAS admits no mirrored stub; see probe_stubs.py

# Direct and thinking variants adjacent, so the within-pair comparison reads across neighbouring bars.
GROUPS = [
    ("Qwen3", ["base_direct", "base_thinking"]),
    ("Ibis0", ["direct_rl", "thinking_rl", "classifier"]),
    ("Frontier", ["gpt56sol", "opus5"]),
]
# Per-bar labels. The group header ("Qwen3" / "Ibis0") already carries the family, so these name only
# the variant -- which makes the first two groups read as mirrored pairs.
SHORT = {
    "base_direct": "non-thinking",
    "base_thinking": "thinking",
    "direct_rl": "non-thinking",
    "thinking_rl": "thinking",
    "classifier": "classifier",
    "gpt56sol": "GPT-5.6-sol",
    "opus5": "Opus 5",
}


def collect(testset, root, boot, domains, per_episode, statistic):
    """{key: (point, lo, hi)} for one probe over the roster."""
    out = {}
    for key, entry in ROSTER.items():
        cells = load_cells(testset, entry.eval_key, root=root, tag="primary")
        per_dom = {d: per_episode(cells[d]) for d in domains if d in cells}
        per_dom = {d: v for d, v in per_dom.items() if len(v)}
        if not per_dom:
            continue
        out[key] = statistic(per_dom, boot)
    return out


def _bars(ax, keys, stats, zero_line):
    for i, key in enumerate(keys):
        if key not in stats:
            continue
        m, lo, hi = stats[key]
        ax.bar(
            i,
            m,
            width=0.62,
            color=plot_color(ROSTER[key]),
            edgecolor=figure_io.GREY["ink"],
            linewidth=0.5,
            yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
            capsize=2.5,
            error_kw={"ecolor": figure_io.GREY["ink"], "elinewidth": 0.9},
        )
    if zero_line:
        ax.axhline(0, color=figure_io.GREY["ref"], lw=0.8, zorder=0)
    ax.set_xlim(-0.62, len(keys) - 0.38)
    ax.set_xticks(range(len(keys)))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# Matched pairs that differ only in whether a reasoning trace is verbalized. Drawn as arrows so the
# within-pair effect is visible rather than asserted: both point left, which is the finding that adding a
# trace lowers first-position bias by about the same amount trained or not.
PAIRS = [("base_direct", "base_thinking"), ("direct_rl", "thinking_rl")]

# SHORT names only the variant, which is unambiguous only under the group titles of the row layout --
# there "non-thinking" appears once per group. With no grouping on screen the Qwen3 and Ibis0 rows
# collide outright (both are "non-thinking"), so the scatter takes the full roster labels. Opus 5 keeps
# its short form: the roster's "Claude Opus 5" is the widest label here and the vendor is not in doubt.
SCATTER_LABEL = {k: e.label for k, e in ROSTER.items()} | {"opus5": "Opus 5"}


# Label offsets in points from the marker, per system. A single rule cannot work here: four of the seven
# sit between 0.26 and 1.55 dB on x, so a uniform top-right offset runs GPT-5.6-sol's label into direct
# RL's marker, and base direct sits close enough to the right edge that its label has to go left. Seven
# points are few enough to place deliberately. `_warn_label_overlaps` is what keeps this honest when the
# numbers move.
LABEL_PLACEMENT = {
    # Lifted into the empty upper-left with a leader line: at 0.26 dB it sits 0.68 dB from direct RL,
    # which is a fifth of an inch at this width, so any label beside the dot crowds that marker even
    # though the two label boxes do not technically overlap.
    # `baseline`, not `bottom`: the leader ends at the text anchor, and a bottom-aligned box reserves
    # descender space this label never uses, so the glyphs floated a couple of points above the line.
    "gpt56sol": (2, 44, "left", "baseline"),
    # These two sit within 0.02 dB of each other on y and 1.9 dB apart on x, and the roster labels are
    # wide enough that 1.9 dB no longer separates them horizontally. So they are split VERTICALLY --
    # direct_rl above its marker, base_thinking below -- rather than pushed apart on x, which at
    # direct_rl's 0.94 dB would drive its label off the left of the axes and into the range annotation.
    "direct_rl": (5, 7, "left", "bottom"),
    "thinking_rl": (0, -11, "center", "top"),
    "opus5": (9, -1, "left", "center"),
    "base_thinking": (0, -11, "center", "top"),
    "classifier": (8, 3, "left", "bottom"),
    "base_direct": (6, 4, "left", "bottom"),
}


def _warn_label_overlaps(fig, ax):
    """Report labels that collide with each other. Nothing clips the canvas when this happens, so the
    overflow check in `figure_io` cannot see it and the figure simply reads wrong."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [(x.get_text(), x.get_window_extent(r)) for x in ax.texts if x.get_text()]
    for i, (na, ba) in enumerate(boxes):
        for nb, bb in boxes[i + 1 :]:
            if ba.overlaps(bb):
                print(f"[belief_revision] !! labels overlap: {na!r} and {nb!r}", file=sys.stderr)


# Stub colours for the schematic. Deliberately not roster colours: these are evidence, not systems, and
# reusing the palette would suggest a correspondence that does not exist.
STUB_GREEN = "#3f8f5a"  # the compelling stub -- supporting, top relevance tier
STUB_RED = "#b5453f"  # its constructed mirror -- equal magnitude, opposite direction
STUB_GREY = "#d8d8d8"  # neutral padding


def _stub(ax, x, y, w, h, color, edge="#ffffff"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={min(w, h) * 0.34}",
            facecolor=color,
            edgecolor=edge,
            linewidth=0.5,
            mutation_aspect=1,
            clip_on=False,
        )
    )


def _schematic(ax, positions=(1, 4, 7, 10), pool=10):
    """What was done to the evidence, and what was computed from it.

    Both probes hold content fixed and vary structure, which is the one thing a scatter of their results
    cannot show. Each block pairs the manipulation with the statistic it produces, so the figure carries
    the definition rather than deferring both to the caption: the top is the same compelling stub walking
    through an otherwise identical context, the bottom is that stub, its constructed mirror, and the pair.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fs = figure_io.FONT["annot"]
    ink, note = figure_io.GREY["ink"], figure_io.GREY["note"]
    x0, gap = 0.19, 0.006
    w = (0.99 - x0 - (pool - 1) * gap) / pool

    def header(y, title, sub, formula, gloss):
        """Block header, returning the y at which the rows may start.

        The subtitle is placed from the MEASURED bottom of the formula, not a fixed offset. A stacked
        formula is about twice the height of a text line -- the position statistic sets a fraction over a
        sum -- so anchoring both to the same baseline puts the formula's descender through the subtitle,
        and by how much depends on which formula it is.
        """
        ax.text(0, y, title, ha="left", va="top", fontsize=fs, color=ink, weight="bold")
        # STIX for the mathtext: a real math font rather than matplotlib's DejaVu Sans default, whose glyphs
        # are noticeably blocky at this size. Serif but Times-like rather than Computer Modern, so it reads as
        # notation without looking like an equation lifted out of the body text.
        # Set per text rather than through rcParams, since mathtext is parsed at draw time and an
        # rc_context around construction would not reach it.
        ft = ax.text(
            0.99, y + 0.014, formula, ha="right", va="top", fontsize=fs + 1.6, color=ink, math_fontfamily="stix"
        )
        bb = ft.get_window_extent(ax.figure.canvas.get_renderer())
        formula_bottom = ax.transAxes.inverted().transform((bb.x0, bb.y0))[1]
        y_sub = min(y - 0.060, formula_bottom - 0.028)
        ax.text(0, y_sub, sub, ha="left", va="top", fontsize=fs, color=note, style="italic")
        ax.text(0.99, y_sub, gloss, ha="right", va="top", fontsize=fs, color=note, math_fontfamily="stix")
        return y_sub - 0.075

    # --- position: one row per condition, the green stub migrating along a fixed context ---
    # Bayes-factor form, matching the appendix. An odds ratio between two conditions is itself a Bayes
    # factor, so the statistic is the mean log BF of the first-position context against each later one.
    # The reference differs between the two blocks, so each states its own.
    y = header(
        1.00,
        "position-1 bias",
        "same stubs, different order",
        r"$\frac{1}{|K|}\sum_{k \in K}\ \ \log \mathrm{BF}_{1:k}$",
        r"$\mathrm{BF}_{1:k}=\mathrm{odds}_{1}/\mathrm{odds}_{k}$",
    )
    h = 0.054
    for i, p_ in enumerate(positions):
        yy = y - i * (h + 0.028) - h
        ax.text(x0 - 0.03, yy + h / 2, f"pos {p_}", ha="right", va="center", fontsize=fs, color=note)
        for s in range(pool):
            _stub(ax, x0 + s * (w + gap), yy, w, h, STUB_GREEN if s == p_ - 1 else STUB_GREY)
        last = yy

    # --- consistency: the stub, its mirror, and the pair ---
    y = header(
        last - 0.135,
        "belief-consistency violation",
        "a stub, its mirror, and both",
        r"$\left|\,\log \mathrm{BF}_{\mathrm{both}}-\log\!\left(\mathrm{BF}_{(+)}\cdot\mathrm{BF}_{(-)}\right)\right|$",
        r"$\mathrm{BF}$ vs. no evidence",
    )
    # `both` sits in its own column beside the two singles rather than under them: it is not a third
    # condition of the same kind, it is the pair whose update the other two are supposed to predict, and
    # stacking it in the same list made it read as one more item.
    bw, bh = w * 1.9, 0.060
    for i, (lab, c) in enumerate([("support", STUB_GREEN), ("refute", STUB_RED)]):
        yy = y - i * (bh + 0.030) - bh
        ax.text(x0 - 0.03, yy + bh / 2, lab, ha="right", va="center", fontsize=fs, color=note)
        _stub(ax, x0, yy, bw, bh, c)
    # Centred on the pair it is built from.
    y_both = y - (bh + 0.030) / 2 - bh
    x_both = 0.60
    # U+225F, "questioned equal to". `\stackrel` and `\overset` do not exist in matplotlib mathtext, and
    # STIXGeneral carries the glyph, so it is set as a character rather than assembled from a `?` over an
    # `=`. It states the probe as a question: does the pair move the belief as far as the two singles
    # predict? That is exactly the residual the formula measures.
    ax.text(
        (x0 + bw + x_both) / 2,
        y_both + bh / 2,
        "\u225f",
        ha="center",
        va="center",
        fontsize=fs + 3.5,
        color=figure_io.GREY["note"],
        fontfamily="STIXGeneral",
    )
    ax.text(x_both - 0.03, y_both + bh / 2, "both", ha="right", va="center", fontsize=fs, color=note)
    for j, c in enumerate([STUB_GREEN, STUB_RED]):
        _stub(ax, x_both + j * (bw + gap * 2), y_both, bw, bh, c)


def render_scatter(pos, con, out):
    """One panel: the two probes against each other, one point per system.

    The row layout ranks each probe separately, so a reader has to hold two orderings in mind to notice
    that the same system is worst on both. Plotting them against each other makes that a position on the
    page: the good corner is bottom-left.

    Consistency violation is on x because it has the wide range (0.26 to 7.5 dB against 1.3 dB for
    position bias), so a panel spanning the text width is the natural shape rather than one that stretches
    the data.

    Both axes linear, so distances are differences.
    """
    import matplotlib

    matplotlib.use("Agg")

    keys = [k for k in pos if k in con]
    # Half text width. The scatter carries seven points and two axes, which does not need a full column,
    # and at 0.5\textwidth it can sit beside prose rather than interrupting it.
    # Full width: the scatter keeps the half-column it needs, and the freed half carries a schematic of
    # what was done to the evidence. The results and the manipulation that produced them belong together.
    W, H = figure_io.TEXT_WIDTH_IN, 2.88
    fig = figure_io.new_figure(H)
    ax = fig.add_axes((0.80 / W, 0.66 / H, 2.46 / W, 1.94 / H))
    sch = fig.add_axes((3.66 / W, 0.30 / H, 2.88 / W, 2.42 / H))
    _schematic(sch)

    for k in keys:
        x, xlo, xhi = con[k]
        y, ylo, yhi = pos[k]
        col = plot_color(ROSTER[k])
        # Intervals as thin crosses. Some are wide -- base direct's consistency runs [6.03, 9.04] -- and a
        # clean scatter would overstate how sharply the systems separate.
        ax.plot([max(xlo, 0), xhi], [y, y], color=col, lw=0.8, alpha=0.45, solid_capstyle="butt", zorder=1)
        ax.plot([x, x], [ylo, yhi], color=col, lw=0.8, alpha=0.45, solid_capstyle="butt", zorder=1)
        ax.plot([x], [y], "o", ms=5.5, color=col, mec=figure_io.GREY["ink"], mew=0.5, zorder=3)

    for a, b in PAIRS:
        if a in pos and b in pos:
            ax.annotate(
                "",
                xy=(con[b][0], pos[b][0]),
                xytext=(con[a][0], pos[a][0]),
                arrowprops=dict(
                    arrowstyle="-|>", color=figure_io.GREY["note"], lw=0.9, shrinkA=7, shrinkB=7, alpha=0.8
                ),
                zorder=2,
            )

    ax.axhline(0, color=figure_io.GREY["ref"], lw=0.9, zorder=0)
    ax.set_xlabel("belief-consistency violation (dB)", fontsize=figure_io.FONT["label"])
    ax.set_ylabel("position-1 bias (dB)", fontsize=figure_io.FONT["label"])
    # A second line per axis saying which way is which. Both quantities are signed departures from zero
    # and neither name says on its own whether more is worse -- "position-1 bias" in particular reads as
    # a magnitude until you notice the axis runs negative.
    ax.xaxis.set_label_coords(0.5, -0.135)
    ax.yaxis.set_label_coords(-0.115, 0.5)
    ax.text(
        0.5,
        -0.225,
        "consistent \u2194 inconsistent",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=figure_io.FONT["annot"],
        color=figure_io.GREY["note"],
        style="italic",
    )
    ax.text(
        -0.185,
        0.5,
        "weak influence \u2194 strong influence",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=figure_io.FONT["annot"],
        color=figure_io.GREY["note"],
        style="italic",
    )
    ax.tick_params(labelsize=figure_io.FONT["tick"])
    ax.set_xlim(left=0)
    ax.margins(x=0.12, y=0.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Limits are final above, because a leader line is placed in data coordinates and would move if they
    # changed afterwards.
    for k in keys:
        dx, dy, ha, va = LABEL_PLACEMENT.get(k, (6, 4, "left", "bottom"))
        xy = (con[k][0], pos[k][0])
        # A label set far from its marker needs a leader, or it reads as belonging to whatever is nearest.
        # The leader is a separate line rather than `annotate(arrowprops=...)`: an annotation's window
        # extent includes its arrow, so an arrow-bearing label measures as a box spanning the whole
        # distance back to its point and `_warn_label_overlaps` reports a collision with everything in
        # between.
        if abs(dx) > 20 or abs(dy) > 20:
            d = ax.transData.transform(xy)
            end = ax.transData.inverted().transform((d[0] + dx * fig.dpi / 72, d[1] + dy * fig.dpi / 72))
            ax.plot([xy[0], end[0]], [xy[1], end[1]], color=figure_io.GREY["rule"], lw=0.6, zorder=0, clip_on=False)
        ax.annotate(
            SCATTER_LABEL[k],
            xy,
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=figure_io.FONT["annot"],
            color=figure_io.GREY["ink"],
        )

    _warn_label_overlaps(fig, ax)
    figure_io.save_figure(fig, out, tag="belief_revision_scatter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position-testset", default="local-position")
    ap.add_argument("--conflict-testset", default="local-conflict")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out-dir", default="outputs", help="directory for the figures")
    ap.add_argument("--out-name", default="belief_revision", help="basename; the rows layout is suffixed")
    ap.add_argument("--boot", type=int, required=True)
    ap.add_argument(
        "--layout",
        choices=["rows", "scatter", "both"],
        default="scatter",
        help="`scatter` is the published layout and the default; `rows` is the earlier grouped-bar "
        "version, kept because it shows each probe's ranking separately",
    )
    args = ap.parse_args()

    root = Path(args.eval_root) if args.eval_root else None
    args.out = Path(args.out_dir) / args.out_name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pos = collect(args.position_testset, root, args.boot, DOMAINS, episode_displacements, first_position_bias)
    con = collect(args.conflict_testset, root, args.boot, CONFLICT_DOMAINS, episode_residuals, macro_ci)
    if not pos or not con:
        raise SystemExit("missing cells for one of the probes; nothing to draw")

    groups = [(t, [k for k in ks if k in pos or k in con]) for t, ks in GROUPS]
    groups = [(t, ks) for t, ks in groups if ks]
    widths_raw = [len(ks) for _, ks in groups]
    span = figure_io.TEXT_WIDTH_IN - 1.02 - 0.16 - 0.34 * (len(groups) - 1)
    widths = [span * w / sum(widths_raw) for w in widths_raw]

    fig = figure_io.new_figure(4.35)
    axes = []
    for r, row_y in enumerate((2.44, 0.86)):
        axes.append(
            figure_io.panel_row(fig, len(groups), y=row_y, h=1.42, left=1.02, right=0.16, gap=0.34, widths=widths)
        )

    for row, (stats, zero, label) in enumerate(
        [(pos, True, "position-1 bias (dB)"), (con, False, "belief-consistency\nviolation (dB)")]
    ):
        lo = min(v[1] for v in stats.values())
        hi = max(v[2] for v in stats.values())
        pad = 0.14 * (hi - lo)
        for ax, (title, keys) in zip(axes[row], groups):
            _bars(ax, keys, stats, zero)
            ax.set_ylim(min(lo - pad, 0) if zero else 0, hi + pad)
            ax.tick_params(axis="y", labelsize=figure_io.FONT["tick"])
            if row == 0:
                ax.set_title(title, fontsize=figure_io.FONT["title"])
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([SHORT[k] for k in keys], fontsize=figure_io.FONT["annot"], rotation=26, ha="right")
        for ax in axes[row][1:]:
            ax.set_yticklabels([])
        axes[row][0].set_ylabel(label, fontsize=figure_io.FONT["label"])

    if args.layout in ("rows", "both"):
        figure_io.save_figure(fig, str(args.out) + "_rows", tag="belief_revision_rows")
    else:
        import matplotlib.pyplot as plt

        plt.close(fig)
    if args.layout in ("scatter", "both"):
        render_scatter(pos, con, args.out)

    # Full roster labels, not SHORT: this table flattens the groups away, and SHORT leans on the group
    # title to disambiguate -- printed flat, Qwen3 and Ibis0 would both read as a bare "non-thinking".
    print(f"\n{'model':22s} {'position-1 bias':>26s}   {'consistency violation':>26s}")
    for _, keys in groups:
        for k in keys:
            p = f"{pos[k][0]:+.2f} [{pos[k][1]:+.2f}, {pos[k][2]:+.2f}]" if k in pos else "--"
            c = f"{con[k][0]:.2f} [{con[k][1]:.2f}, {con[k][2]:.2f}]" if k in con else "--"
            print(f"  {ROSTER[k].label:20s} {p:>26s}   {c:>26s}")


if __name__ == "__main__":
    main()
