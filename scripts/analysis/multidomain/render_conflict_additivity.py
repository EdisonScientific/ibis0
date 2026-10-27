"""Coherence of belief revision under conflicting evidence, macro-averaged over domains.

Each revealed stub induces a Bayes factor relative to the no-evidence prior, BF_x = odds(x)/odds(none).
A coherent updater treats two conditionally independent measurements multiplicatively, so

    BF_both = BF_support x BF_refute        i.e. in decibans the log Bayes factors ADD.

We report the magnitude of the departure,

    |log BF_both - (log BF_support + log BF_refute)|   in decibans, 0 = exactly multiplicative,

which is the model judged against *its own* single-stub responses rather than against an external
standard. A model that barely moves on either stub is not rewarded for it: the prediction scales with its
own updates.

**Both orderings are averaged into `both`.** The pair is presented support-first and refute-first and the
two are averaged, so a recency or primacy effect cancels instead of inflating the residual. Order is a
separate probe (`render_woe_position.py`); mixing the two would attribute one to the other.

**Tahoe and DepMap only.** The mirror is an equally compelling measurement of the same experiment pointing
the other way, and GWAS evidence has no such category: every stub is a real published association, and a
catalog holds no record asserting a variant fails to associate with a trait. See `probe_stubs.py`. This is
the same reason `render_woe_structure.py` omits GWAS from its directional figure.

Saved through `figure_io`, which disables `bbox_inches="tight"` on purpose: a tight box makes the PDF a
size LaTeX then rescales.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS = next(p for p in _HERE.parents if p.name == "scripts")
sys.path.insert(0, str(_SCRIPTS / "lib"))
sys.path.insert(0, str(_HERE))

import boot_cache  # noqa: E402
import figure_io  # noqa: E402
from multidomain_cells import load_cells  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

DECIBAN = 10.0 / np.log(10.0)
EPS = 1e-3  # logit clip, matching the weight-of-evidence convention
CONDS = ("none", "support", "refute", "both_sfirst", "both_rfirst")
DOMAINS = ("tahoe", "depmap")


def _deciban(p: float) -> float:
    p = min(max(float(p), EPS), 1 - EPS)
    return DECIBAN * np.log(p / (1 - p))


def episode_residuals(rows: list[dict]) -> list[float]:
    """Signed additivity residual per episode, in decibans.

    An episode contributes only when all five conditions parsed: the residual is a contrast across them,
    so a partial episode is not a weaker observation, it is not an observation.
    """
    by: dict[str, dict[str, float]] = {}
    for r in rows:
        base, cond = str(r["episode_id"]).rsplit("__", 1)
        by.setdefault(base, {})[cond] = r["parsed_prob"]
    out = []
    for v in by.values():
        if not all(c in v for c in CONDS):
            continue
        lo = {c: _deciban(v[c]) for c in CONDS}
        a = lo["support"] - lo["none"]
        b = lo["refute"] - lo["none"]
        both = 0.5 * (lo["both_sfirst"] + lo["both_rfirst"])
        out.append(both - (lo["none"] + a + b))
    return out


def macro_ci(per_domain: dict[str, list[float]], n_boot: int, seed: int = 0):
    """Memoised wrapper; the arithmetic is in `_macro_ci_uncached`. See scripts/lib/boot_cache.py.

    These probe bootstraps were the only uncached ones in the pipeline, and they run the highest draw
    count in it, so every re-render recomputed them from scratch. The key covers the resampled data, the
    draw count, the seed, and the source of this file -- so any change that could move the numbers misses
    the cache rather than serving a stale interval.

    Always returns a tuple: `json` round-trips one to a list, and a value whose TYPE depended on whether
    the cache was warm would be a genuinely nasty bug to chase.
    """
    v = boot_cache.cached(
        "probe.conflict_macro_ci",
        {"data": boot_cache.fingerprint_records(per_domain), "n_boot": n_boot, "seed": seed},
        lambda: _macro_ci_uncached(per_domain, n_boot, seed),
        impl_files=[__file__, boot_cache.__file__],
    )
    return None if v is None else tuple(v)


def _macro_ci_uncached(per_domain: dict[str, list[float]], n_boot: int, seed: int = 0):
    """(point, lo, hi) for |macro-average residual|, by within-domain episode bootstrap.

    Episodes are resampled inside each domain and the domain means are then averaged with equal weight,
    so neither domain's episode count sets the aggregate. The absolute value is taken AFTER averaging,
    on each draw, because |mean| and mean|.| are different quantities and the claim is about the former.
    """
    doms = [d for d in DOMAINS if per_domain.get(d)]
    if not doms:
        return None
    rng = np.random.default_rng(seed)
    arrs = {d: np.asarray(per_domain[d], dtype=float) for d in doms}
    point = abs(float(np.mean([arrs[d].mean() for d in doms])))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        means = [arrs[d][rng.integers(0, len(arrs[d]), len(arrs[d]))].mean() for d in doms]
        draws[b] = abs(float(np.mean(means)))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# Three groups, matching the position figure. Grouping separates the training regime from the ranking,
# so a reader can see both that trained and frontier systems combine conflicting evidence more coherently
# than the untrained base and where each system falls inside its own group.
PANELS = [
    ("Qwen3", ["base_direct", "base_thinking"]),
    ("Ibis0", ["classifier", "direct_rl", "thinking_rl"]),
    ("Frontier", ["gpt56sol", "opus5"]),
]


def draw_panels(stats, out):
    """One panel per group, bars ascending within group, shared y so heights compare across panels."""
    fig = figure_io.new_figure(3.05)
    present = [(title, [k for k in keys if k in stats]) for title, keys in PANELS]
    present = [(ti, ks) for ti, ks in present if ks]
    widths = [max(len(ks), 1) for _, ks in present]
    span = figure_io.TEXT_WIDTH_IN - 1.02 - 0.16 - 0.34 * (len(present) - 1)
    widths = [span * w / sum(widths) for w in widths]
    axes = figure_io.panel_row(fig, len(present), y=1.00, h=1.80, left=1.02, right=0.16, gap=0.34, widths=widths)
    ymax = max(hi for _, _, hi in stats.values()) * 1.18
    for ax, (title, keys) in zip(axes, present):
        keys = sorted(keys, key=lambda k: stats[k][0])  # ascending: most coherent leftmost
        for i, key in enumerate(keys):
            m, lo, hi = stats[key]
            e = ROSTER[key]
            ax.bar(
                i,
                m,
                width=0.60,
                color=plot_color(e),
                edgecolor=figure_io.GREY["ink"],
                linewidth=0.5,
                yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
                capsize=3,
                error_kw={"ecolor": figure_io.GREY["ink"], "elinewidth": 0.9},
            )
            ax.text(
                i,
                hi + 0.10,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=figure_io.FONT["annot"],
                color=figure_io.GREY["ink"],
            )
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.62, len(keys) - 0.38)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([ROSTER[k].label for k in keys], fontsize=figure_io.FONT["annot"], rotation=28, ha="right")
        ax.set_title(title, fontsize=figure_io.FONT["title"])
        ax.tick_params(axis="y", labelsize=figure_io.FONT["tick"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[0].set_ylabel("belief-consistency\nviolation (dB)", fontsize=figure_io.FONT["label"])
    figure_io.save_figure(fig, out, tag="conflict_additivity_panels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="local-conflict", help="eval-tree slug the probe wrote under")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--out-dir", default="outputs", help="directory for the figures")
    ap.add_argument("--out-name", default="conflict_additivity", help="basename; layout variants are suffixed")
    ap.add_argument("--boot", type=int, required=True)
    ap.add_argument("--layout", choices=["single", "panels", "both"], default="both")
    args = ap.parse_args()
    args.out = Path(args.out_dir) / args.out_name
    args.out.parent.mkdir(parents=True, exist_ok=True)

    root = Path(args.eval_root) if args.eval_root else None
    stats, counts = {}, {}
    for key in ORDER:
        entry = ROSTER[key]
        cells = load_cells(args.testset, entry.eval_key, root=root, tag="primary")
        per_dom = {d: episode_residuals(cells[d]) for d in DOMAINS if d in cells}
        per_dom = {d: v for d, v in per_dom.items() if v}
        if not per_dom:
            print(f"  [skip] {entry.label}: no cells under {args.testset}/{entry.eval_key}")
            continue
        ci = macro_ci(per_dom, args.boot)
        stats[key] = ci
        counts[key] = {d: len(v) for d, v in per_dom.items()}

    if not stats:
        raise SystemExit(f"no roster row had cells under {args.testset}; nothing to draw")

    if args.layout in ("panels", "both"):
        draw_panels(stats, str(args.out) + "_panels")

    order = sorted(stats, key=lambda k: stats[k][0])  # most coherent first
    fig = figure_io.new_figure(3.1)
    (ax,) = figure_io.panel_row(fig, 1, y=1.02, h=1.72, left=1.05, right=0.14)
    for i, key in enumerate(order):
        m, lo, hi = stats[key]
        e = ROSTER[key]
        ax.bar(
            i,
            m,
            width=0.62,
            color=e.color,
            edgecolor=figure_io.GREY["ink"],
            linewidth=0.5,
            yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
            capsize=3,
            error_kw={"ecolor": figure_io.GREY["ink"], "elinewidth": 0.9},
        )
        ax.text(
            i,
            hi + 0.04,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=figure_io.FONT["annot"],
            color=figure_io.GREY["ink"],
        )
    ax.axhline(0, color=figure_io.GREY["ref"], lw=0.8)
    # Headroom for the value labels, set in data coords so they cannot overhang the canvas.
    ax.set_ylim(0, max(stats[k][2] for k in order) * 1.20)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ROSTER[k].label for k in order], fontsize=figure_io.FONT["tick"], rotation=20, ha="right")
    ax.tick_params(axis="y", labelsize=figure_io.FONT["tick"])
    ax.set_ylabel("belief-consistency violation (dB)", fontsize=figure_io.FONT["label"])
    ax.margins(x=0.06)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    figure_io.save_figure(fig, args.out, tag="conflict_additivity")
    print(f"\n{'model':22s} {'|residual| dB':>14s}   95% CI            episodes")
    for k in order:
        m, lo, hi = stats[k]
        n = "  ".join(f"{d}={c}" for d, c in counts[k].items())
        print(f"  {ROSTER[k].label:20s} {m:12.3f}   [{lo:6.3f}, {hi:6.3f}]   {n}")


if __name__ == "__main__":
    main()
