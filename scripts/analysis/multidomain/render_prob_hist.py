"""Predicted-probability distributions on the published multi-domain TEST set.

A grid of filled histograms
with a boundary-reflected KDE, a dashed rule at 0.5, and n / mean / sd in each panel title. It answers a
question none of the scalar metrics do — *where* a model puts its probability mass. A model can look
mediocre on Brier because it is uniformly shifted, or because it is bimodal at 0 and 1, and only the
distribution separates those.

**What is pooled.** The published `primary` config carries ONE k per episode, so each episode
contributes exactly one forecast under the default `--tag primary`: the distribution of *final* beliefs,
not of beliefs in progress. `--tag 'ctx*'` reads the dense `sensitivity` sweep instead and pools all 17
forecasts per episode, which is the true pooled-over-deliberation figure.

**`--tag` is what selects the config, NOT `--testset`.** Both configs share a `<testset>` slug — it keys
on repo + revision, not config — so they share a result directory and are told apart only by each cell's
tag. Pointing `--testset` at `sensitivity/` while leaving the tag alone therefore re-reads the `primary`
cells and mislabels them as the sensitivity result, which is indistinguishable from a genuine null result.
`--testset` still selects the *directory* (revision), so it must stay consistent with the tag.

**Domain weighting.** Domains are pooled with EQUAL WEIGHT PER DOMAIN rather than by episode count. With
every episode parsed the two coincide, but they diverge exactly when a row loses episodes to parse
failures, and a figure whose weighting silently depends on a model's failure
rate is not comparable across rows. Weighting explicitly makes the panel macro by construction, matching
`macro_bootstrap` and the rest of this directory.

Default panels: GPT-5.6-sol, Claude Opus 5, Qwen base (direct) and the classifier. `--keys` overrides.
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

import eval_paths  # noqa: E402
from figure_io import FONT, GREY, new_figure, panel_grid, save_figure  # noqa: E402
from multidomain_cells import DOMAINS, load_cells  # noqa: E402
from roster_palette import ROSTER, plot_color  # noqa: E402

# Four representative systems: both frontier models, the untrained base and the classifier. Not in roster
# order, so the two frontier panels sit side by side.
DEFAULT_KEYS = ("gpt56sol", "opus5", "base_direct", "classifier")


def weighted_probs(cells: dict[str, list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """(probs, weights) pooling every domain with equal total weight.

    Each domain's episodes share a total weight of 1/n_domains, so a domain contributes the same mass
    regardless of how many of its episodes parsed. Weights (not resampling) because a distribution plot
    should show the actual forecasts, not a resampled surrogate of them.
    """
    ps, ws = [], []
    n_dom = len(cells)
    for rows in cells.values():
        p = np.array([r["parsed_prob"] for r in rows], dtype=float)
        ps.append(p)
        ws.append(np.full(len(p), 1.0 / (n_dom * len(p))))
    return np.clip(np.concatenate(ps), 0.0, 1.0), np.concatenate(ws)


def _wmean_wsd(p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = float(np.sum(w * p) / np.sum(w))
    return m, float(np.sqrt(np.sum(w * (p - m) ** 2) / np.sum(w)))


def render(series: list[tuple[str, np.ndarray, np.ndarray]], out_base: Path, bw: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from scipy.stats import gaussian_kde

    n = len(series)
    ncol = 2 if n <= 4 else 3
    nrow = int(np.ceil(n / ncol))
    # Absolute inches at TEXT_WIDTH_IN, so LaTeX places the figure at `width=\textwidth` without rescaling
    # and the declared font sizes are the printed ones.
    L_, R_, GAP_, RGAP = 0.56, 0.10, 0.30, 0.44
    PH, TOP, BOT = 1.15, 0.30, 0.44
    figh = TOP + nrow * PH + (nrow - 1) * RGAP + BOT
    fig = new_figure(figh)
    flat = panel_grid(fig, nrow, ncol, y=BOT, h=PH, row_gap=RGAP, left=L_, right=R_, gap=GAP_)
    axes = np.array(flat, dtype=object).reshape(nrow, ncol)
    bins = np.linspace(0, 1, 31)
    xs = np.linspace(0, 1, 400)

    for ax, (key, p, w) in zip(flat, series):
        e = ROSTER[key]
        col = plot_color(e)
        ax.hist(p, bins=bins, weights=w, density=True, color=col, alpha=0.35, edgecolor="white", linewidth=0.4)
        # Reflect the kernel at both boundaries: probabilities live on [0, 1], and an unreflected KDE
        # leaks mass past the edges and understates density exactly where these models pile up.
        kde = gaussian_kde(p, bw_method=bw, weights=w)
        dens = kde(xs) + kde(-xs) + kde(2 - xs)
        ax.plot(xs, dens, color=col, lw=1.6)
        ax.tick_params(labelsize=FONT["tick"])
        ax.fill_between(xs, dens, color=col, alpha=0.18)
        ax.axvline(0.5, ls="--", lw=1, color=GREY["ref"], zorder=0)
        m, sd = _wmean_wsd(p, w)
        ax.set_title(f"{e.label}  (n={len(p)}, mean {m:.2f}, sd {sd:.2f})", fontsize=FONT["annot"])
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.25)
    for ax in flat[n:]:
        ax.set_visible(False)

    # Label only the outer edges, but on the LAST visible row of each column -- with a partly filled grid
    # the bottom row of `axes` can be hidden, which would drop the x labels entirely.
    for c in range(ncol):
        col_axes = [axes[r][c] for r in range(nrow) if axes[r][c].get_visible()]
        if col_axes:
            col_axes[-1].set_xlabel(r"Model $p(\theta)$", fontsize=FONT["label"])
    for r in range(nrow):
        if axes[r][0].get_visible():
            axes[r][0].set_ylabel("density", fontsize=FONT["label"])

    save_figure(fig, out_base, tag="prob-hist", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None, help="override the eval tree root")
    ap.add_argument(
        "--keys",
        default=",".join(DEFAULT_KEYS),
        help=f"comma list of roster keys, one panel each (default: {','.join(DEFAULT_KEYS)})",
    )
    ap.add_argument(
        "--tag",
        default="primary",
        help="which cells to read: 'primary' (one final forecast per episode, the default), a single "
        "'ctx<k>', or 'ctx*' to pool every k of the sensitivity sweep (pooled-over-deliberation). "
        "This — not --testset — is what selects the config; see the module docstring.",
    )
    ap.add_argument("--out-dir", default="outputs/prob_hist")
    ap.add_argument("--out-name", default=None, help="basename of the figure (default: keyed to --tag)")
    ap.add_argument("--bw", type=float, default=0.2, help="gaussian_kde bandwidth (matches the original)")
    args = ap.parse_args()

    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if unknown := [k for k in keys if k not in ROSTER]:
        raise SystemExit(f"unknown roster keys: {unknown} (known: {list(ROSTER)})")

    series, stats = [], []
    for key in keys:
        cells = load_cells(slug, ROSTER[key].eval_key, root=root, tag=args.tag)
        if not cells:
            print(f"[prob-hist] skip {key}: no cells under {ROSTER[key].eval_key}", file=sys.stderr)
            continue
        missing = [d for d in DOMAINS if d not in cells]
        if missing:
            print(f"[prob-hist] {ROSTER[key].label}: missing {missing} — panel covers the rest", file=sys.stderr)
        p, w = weighted_probs(cells)
        series.append((key, p, w))
        m, sd = _wmean_wsd(p, w)
        stats.append(
            {
                "key": key,
                "label": ROSTER[key].label,
                "n": int(len(p)),
                "per_domain_n": {d: len(v) for d, v in sorted(cells.items())},
                "weighted_mean": m,
                "weighted_sd": sd,
            }
        )
    if not series:
        raise SystemExit(f"no results found under {eval_paths.result_dir(slug, '', root=root).parent}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Name the artifacts after the tag, so the pooled-over-deliberation figure cannot overwrite the
    # final-belief one it exists to be compared against.
    pooled = args.tag == "ctx*"
    base = args.out_name or ("prob_hist_pooled_deliberation" if pooled else "prob_hist_pooled")
    (out_dir / f"{base}_stats.json").write_text(
        # `--tag` selects the config, not `--testset` (see the module docstring), so recording the
        # bare --testset value labels a sensitivity render as primary. Report the config the tag
        # actually read.
        json.dumps(
            {
                "tag": args.tag,
                "testset": args.testset.replace("/primary/", "/sensitivity/")
                if args.tag.startswith("ctx")
                else args.testset,
                "rows": stats,
            },
            indent=2,
        )
    )

    render(
        series,
        out_dir / base,
        args.bw,
    )
    for s in stats:
        print(f"  {s['label']:24s} n={s['n']:5d} mean={s['weighted_mean']:.3f} sd={s['weighted_sd']:.3f}")
    print(f"[prob-hist] stats -> {out_dir / f'{base}_stats.json'}")


if __name__ == "__main__":
    main()
