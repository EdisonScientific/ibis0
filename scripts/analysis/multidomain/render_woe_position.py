"""Does where a piece of evidence sits change what a model concludes from it?

The same compelling stub is placed at position 1, 4, 7 or 10 of an otherwise identical context, with the
same padding in the same order around it. **Every condition therefore shows the model the same multiset
of evidence**, and only the order differs, so an order-invariant reasoner must report the same
probability in all four. What is plotted is the belief itself, in decibans relative to each episode's own
mean across positions:

    displacement(p) = logit(belief at position p) - mean_p' logit(belief at position p')

Centring within episode removes the episode's difficulty, which is a nuisance here: episodes differ
enormously in how compelling their probe stub is, and that variance would otherwise swamp the position
effect. A flat line at zero is order invariance. A downward slope is primacy (early evidence counts for
more), an upward slope is recency.

**Why permute rather than grow the context.** Reading the probe's weight of evidence at position p as
logit(P@k=p) - logit(P@k=p-1) needs each context scored at two lengths and confounds position with how
much evidence has accumulated by then. Holding the
evidence set fixed and permuting it is the cleaner test, and it is the one the section's own premise asks
for: two readers given the same evidence in different orders should end up believing the same thing.

**Pool of ten, not sixteen.** DepMap episodes carry only about eleven stubs that can serve as padding, and
a pool of sixteen leaves 14 eligible episodes out of 2,000. At ten, all three domains clear a thousand.
See `build_probe_sets.py`.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = next(p for p in _HERE.parents if p.name == "scripts")
sys.path.insert(0, str(_SCRIPTS / "lib"))
sys.path.insert(0, str(_HERE))

import boot_cache  # noqa: E402
import figure_io  # noqa: E402
from multidomain_cells import DOMAINS, load_cells  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

DECIBAN = 10.0 / np.log(10.0)
EPS = 1e-3


def _deciban(p: float) -> float:
    p = min(max(float(p), EPS), 1 - EPS)
    return DECIBAN * np.log(p / (1 - p))


def episode_displacements(rows: list[dict]) -> dict[int, list[float]]:
    """{position: [within-episode centred displacement, ...]}.

    Only episodes with every position parsed contribute, for the same reason as the conflict probe: the
    quantity is a contrast across conditions, so a partial episode would shift the mean by dropping one
    arm rather than by showing an order effect.
    """
    by: dict[str, dict[int, float]] = {}
    for r in rows:
        m = re.fullmatch(r"(.+)__pos(\d+)", str(r["episode_id"]))
        if m:
            by.setdefault(m.group(1), {})[int(m.group(2))] = r["parsed_prob"]
    if not by:
        return {}
    positions = sorted({p for v in by.values() for p in v})
    out: dict[int, list[float]] = {p: [] for p in positions}
    for v in by.values():
        if len(v) != len(positions):
            continue
        lo = {p: _deciban(v[p]) for p in positions}
        centre = float(np.mean(list(lo.values())))
        for p in positions:
            out[p].append(lo[p] - centre)
    return {p: xs for p, xs in out.items() if xs}


def macro_curve(per_domain: dict[str, dict[int, list[float]]], n_boot: int, seed: int = 0):
    """(positions, point, lo, hi) for the macro-averaged displacement curve.

    Episodes resample within domain; domains average with equal weight. Resampling is done once per draw
    per domain and reused across positions, so the curve's points stay correlated the way the underlying
    episodes are -- resampling independently per position would produce intervals that imply the shape is
    noisier than it is.
    """
    doms = [d for d in DOMAINS if per_domain.get(d)]
    if not doms:
        return None
    positions = sorted(per_domain[doms[0]].keys())
    arrs = {d: np.array([per_domain[d][p] for p in positions], dtype=float) for d in doms}  # (P, n_ep)
    point = np.mean([arrs[d].mean(axis=1) for d in doms], axis=0)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(positions)))
    for b in range(n_boot):
        per = []
        for d in doms:
            idx = rng.integers(0, arrs[d].shape[1], arrs[d].shape[1])
            per.append(arrs[d][:, idx].mean(axis=1))
        draws[b] = np.mean(per, axis=0)
    return positions, point, np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def first_position_bias(per_domain, n_boot: int, seed: int = 0):
    """Memoised wrapper; the arithmetic is in `_first_position_bias_uncached`. See scripts/lib/boot_cache.py.

    These probe bootstraps were the only uncached ones in the pipeline, and they run the highest draw
    count in it, so every re-render recomputed them from scratch. The key covers the resampled data, the
    draw count, the seed, and the source of this file -- so any change that could move the numbers misses
    the cache rather than serving a stale interval.

    Always returns a tuple: `json` round-trips one to a list, and a value whose TYPE depended on whether
    the cache was warm would be a genuinely nasty bug to chase.
    """
    v = boot_cache.cached(
        "probe.first_position_bias",
        {"data": boot_cache.fingerprint_records(per_domain), "n_boot": n_boot, "seed": seed},
        lambda: _first_position_bias_uncached(per_domain, n_boot, seed),
        impl_files=[__file__, boot_cache.__file__],
    )
    return None if v is None else tuple(v)


def _first_position_bias_uncached(per_domain, n_boot: int, seed: int = 0):
    """(point, lo, hi) for displacement at the first position minus the mean of all later ones, in dB.

    NOT the first-to-last contrast, and not the spread. The endpoint contrast is blind to curve shape:
    direct RL scores -0.00 on it because position 10 happens to be as elevated as position 1, cancelling
    a first-position effect that is real and separated from zero on this statistic. Spread has the
    opposite problem, being bounded below by zero and growing with the number of sampled positions even
    under noise.

    Within-episode centring cancels in any contrast of positions, so this is a raw within-episode logit
    difference and does not depend on which positions were sampled for the centring.
    """
    doms = [d for d in DOMAINS if per_domain.get(d)]
    positions = sorted(per_domain[doms[0]].keys())
    first, rest = positions[0], positions[1:]
    arrs = {}
    for d in doms:
        v = {p: np.array(per_domain[d][p]) for p in positions}
        arrs[d] = v[first] - sum(v[p] for p in rest) / len(rest)
    point = float(np.mean([a.mean() for a in arrs.values()]))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arrs.values()])
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_table(contrasts, counts, out_path):
    """LaTeX table of first-position bias, one row per system. Same conventions as render_master_table:
    generated, not hand-edited, booktabs, systems in roster order."""
    n_ep = sum(next(iter(counts.values())).values()) if counts else 0
    lines = [
        "% Generated by scripts/analysis/multidomain/render_woe_position.py -- do not edit by hand.",
        "% Requires \\usepackage{booktabs} and \\usepackage{xcolor}.",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Model & Position-1 bias (dB) & 95\\% CI \\\\",
        "\\midrule",
    ]
    # Sorted by the reported statistic, not roster order: the table's job is to show where each system
    # falls on this one axis, and ranking makes that readable at a glance. The figure keeps roster
    # grouping, since its panels encode a different comparison.
    for key in sorted(contrasts, key=lambda k: -contrasts[k][0]):
        c, lo, hi = contrasts[key]
        detached = not (lo <= 0 <= hi)
        val = f"\\textbf{{{c:+.2f}}}" if detached else f"{c:+.2f}"
        # Interval in black!45, matching \mtstack and \tmstack. Here the CI has its own column rather
        # than sitting under the point estimate, so the grey is doing less work -- but a reader moving
        # between these appendices should not have to re-learn what a grey number means.
        ci = f"\\textcolor{{black!45}}{{$[{lo:+.2f}, {hi:+.2f}]$}}"
        lines.append(f"{ROSTER[key].label} & {val} & {ci} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{\\textbf{Dependence of the stated belief on the first evidence position.} "
        "Stated belief at position $1$ minus the mean over positions $4$, $7$ and $10$, in decibans, "
        f"macro-averaged over the three domains with ${n_ep // 3 if n_ep else 0}$ episodes per domain. "
        "Positive values indicate that the same stub moves the belief further when it appears first. "
        "Intervals are $95\\%$ episode bootstrap intervals resampled within domain; bold marks an "
        "interval excluding zero.}",
        "\\label{tab:position-bias}",
        "\\end{table}",
        "",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))
    print(f"[woe_position] -> {out_path}")


# Matched-pair panels: each group holds systems that differ in whether they verbalize a reasoning trace,
# so a within-group comparison isolates that difference from the training recipe and the model family.
PANELS = [
    ("Qwen3", ["base_direct", "base_thinking"]),
    ("Ibis0", ["classifier", "direct_rl", "thinking_rl"]),
    ("Frontier", ["gpt56sol", "opus5"]),
]


def draw_panels(curves, contrasts, out, n_ep_note):
    """Three panels sharing a y axis, one per matched group, each on its own axes."""
    fig = figure_io.new_figure(2.95)
    axes = figure_io.panel_row(fig, 3, y=0.66, h=1.92, left=0.82, right=0.16, gap=0.42)
    ylo = min(v[2].min() for v in curves.values())
    yhi = max(v[3].max() for v in curves.values())
    pad = 0.12 * (yhi - ylo)
    for ax, (title, keys) in zip(axes, PANELS):
        for key in keys:
            if key not in curves:
                continue
            pos, pt, lo, hi = curves[key]
            e = ROSTER[key]
            col = plot_color(e)
            c = contrasts[key][0]
            ax.plot(pos, pt, color=col, lw=1.5, marker="o", ms=2.8, label=f"{e.label}  ({c:+.2f})")
            ax.fill_between(pos, lo, hi, color=col, alpha=0.13, linewidth=0)
        ax.axhline(0, color=figure_io.GREY["ref"], lw=0.9, zorder=0)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_xticks(sorted(next(iter(curves.values()))[0]))
        ax.tick_params(labelsize=figure_io.FONT["tick"])
        ax.set_title(title, fontsize=figure_io.FONT["title"])
        ax.set_xlabel("position", fontsize=figure_io.FONT["label"])
        ax.legend(loc="upper right", frameon=False, fontsize=figure_io.FONT["annot"], handlelength=1.2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("belief displacement vs.\nepisode mean (dB)", fontsize=figure_io.FONT["label"])
    figure_io.save_figure(fig, out, tag="woe_position_panels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="local-position")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out-dir", default="outputs", help="directory for the figures")
    ap.add_argument("--out-name", default="woe_position", help="basename; layout variants are suffixed")
    ap.add_argument("--boot", type=int, required=True)
    ap.add_argument("--layout", choices=["single", "panels", "both"], default="both")
    ap.add_argument("--table", default="", help="LaTeX table output; defaults to <out-dir>/position_bias.tex")
    args = ap.parse_args()
    args.out = Path(args.out_dir) / args.out_name
    args.table = args.table or str(Path(args.out_dir) / "position_bias.tex")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    root = Path(args.eval_root) if args.eval_root else None
    curves, contrasts, counts = {}, {}, {}
    for key in ORDER:
        entry = ROSTER[key]
        cells = load_cells(args.testset, entry.eval_key, root=root, tag="primary")
        per_dom = {d: episode_displacements(cells[d]) for d in DOMAINS if d in cells}
        per_dom = {d: v for d, v in per_dom.items() if v}
        if not per_dom:
            print(f"  [skip] {entry.label}: no cells under {args.testset}/{entry.eval_key}")
            continue
        got = macro_curve(per_dom, args.boot)
        if got is None:
            continue
        curves[key] = got
        contrasts[key] = first_position_bias(per_dom, args.boot)
        counts[key] = {d: len(next(iter(v.values()))) for d, v in per_dom.items()}

    if not curves:
        raise SystemExit(f"no roster row had cells under {args.testset}; nothing to draw")

    if args.table:
        write_table(contrasts, counts, args.table)

    if args.layout in ("panels", "both"):
        draw_panels(curves, contrasts, str(args.out) + "_panels", len(counts))
    if args.layout == "panels":
        _report(curves, contrasts)
        return

    fig = figure_io.new_figure(3.2)
    (ax,) = figure_io.panel_row(fig, 1, y=0.62, h=2.30, left=0.95, right=1.55)
    for key in ORDER:
        if key not in curves:
            continue
        pos, pt, lo, hi = curves[key]
        e = ROSTER[key]
        # SOLID for every row, matching render_reliability / render_dose_response. `roster_palette`
        # declares a dash for the same-hue variant (base-thinking) and for the two frontier rows, but
        # `plot_color` already separates the shared hue by shade -- so dashing it too double-encodes
        # one distinction, and on the frontier rows, which have unique colours, it encodes nothing.
        col = plot_color(e)
        ax.plot(pos, pt, color=col, lw=1.5, marker="o", ms=2.8, label=e.label)
        ax.fill_between(pos, lo, hi, color=col, alpha=0.13, linewidth=0)
    ax.axhline(0, color=figure_io.GREY["ref"], lw=0.9, zorder=0)
    ax.text(
        ax.get_xlim()[1],
        0,
        " order-invariant",
        va="center",
        ha="left",
        fontsize=figure_io.FONT["annot"],
        color=figure_io.GREY["note"],
    )
    ax.set_xlabel("position of the probe stub in the context", fontsize=figure_io.FONT["label"])
    ax.set_ylabel("belief displacement vs. this episode's mean\n(decibans)", fontsize=figure_io.FONT["label"])
    ax.set_xticks(sorted(next(iter(curves.values()))[0]))
    ax.tick_params(labelsize=figure_io.FONT["tick"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=figure_io.FONT["legend"],
        handlelength=1.6,
    )

    figure_io.save_figure(fig, args.out, tag="woe_position")
    _report(curves, contrasts)


def _report(curves, contrasts):
    positions = next(iter(curves.values()))[0]
    head = "  ".join(f"{'pos ' + str(p):>8s}" for p in positions)
    print(f"\n{'model':22s} {head}     pos1 vs later (95% CI)")
    for key in ORDER:
        if key not in curves:
            continue
        _, pt, _, _ = curves[key]
        c, lo, hi = contrasts[key]
        star = "" if lo <= 0 <= hi else "  *"
        print(
            f"  {ROSTER[key].label:20s} "
            + "  ".join(f"{v:+8.2f}" for v in pt)
            + f"   {c:+6.2f} [{lo:+.2f}, {hi:+.2f}]{star}"
        )
    print("\n  * interval excludes zero; positive = the first position counts for more")


if __name__ == "__main__":
    main()
