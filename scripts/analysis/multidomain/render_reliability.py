"""Reliability diagrams for the roster on the published multi-domain TEST set.

One row of panels -- the three domains, then the macro-average -- with every roster row overlaid,
dashed diagonal = perfect calibration. This is the figure that shows *how* a model is miscalibrated, which
the scalar columns in the master table cannot: two rows with the same Brier can be over-confident at the
top of the range and under-confident at the bottom, or uniformly shifted, and only the curve distinguishes
those. Every panel is square, so deviation from the diagonal is comparable by eye across all of them.

Deliberately shares three things rather than reimplementing them, so this figure can never disagree with
the master table about what was scored or what it is called:

- `multidomain_cells.load_cells` — identical episode selection, including the same exclusion of
  unparsed episodes.
- `render_calibration_report.reliability_curve_ci` — the established curve: EQUAL-MASS (quantile) bins,
  so bins are wide where predictions are sparse rather than silently empty, with a per-bin bootstrap CI
  on the empirical frequency that widens where a bin holds few episodes.
- `roster_palette.plot_color` — one colour per row, with same-hue variants separated by shade rather than
  by the palette's `linestyle`. Dash is reserved for the perfect-calibration reference here, so a dashed
  model curve would compete with the line it is being compared against.

Equal-mass binning is the reason the x-positions differ between models: each curve's bins hold ~n/B of
*that model's* forecasts, so a model that crowds its predictions into a narrow band gets bins spanning
that band rather than a row of empty ones. That is the honest presentation, but it means the curves are
not sampled at shared x — do not read vertical differences at a given x as a paired comparison.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import display_names as dn  # noqa: E402
import eval_paths  # noqa: E402
from figure_io import FONT, GREY, TEXT_WIDTH_IN, new_figure, panel_row, save_figure  # noqa: E402
from multidomain_cells import DOMAINS, load_cells  # noqa: E402
from render_calibration_report import N_BINS, reliability_curve_ci  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

MACRO_KEY = "__macro__"


def macro_curve(
    cells: dict[str, list[dict]],
    n_bins: int,
    n_boot: int,
    min_domains: int = 2,
    min_count: int = 5,
    seed: int = 0,
) -> list[tuple]:
    """Macro-averaged reliability curve: per-domain bin frequency, then equal-weight mean over domains.

    Consistency with the grouped-metrics figure is the point. `macro_bootstrap` defines the macro as
    "statistic per domain, then nanmean over domains", with the bootstrap resampling episodes WITHIN
    domain; this does the same thing with "empirical frequency in bin k" as the statistic. Pooling all
    episodes and binning once would instead give a MICRO average -- each bin's frequency weighted by
    however many episodes of each domain happened to land in it -- which is exactly what macro-averaging
    exists to avoid, and would let the largest-|p| domain dominate the curve's shape.

    Bins need SHARED edges across domains, because per-domain equal-mass bins put each domain's points at
    different x and pointwise averaging would then compare unlike positions. Edges come from quantiles of
    the pooled predictions, so they are populated overall, and each domain is then binned into them.

    Two guards, because shared edges do not guarantee per-domain occupancy (at 15 bins one Tahoe bin is
    empty and one DepMap bin holds 2 episodes):
      - a (domain, bin) cell with < `min_count` episodes contributes nan rather than a 0/1-ish estimate;
      - a bin with fewer than `min_domains` contributing domains is dropped entirely, since a "macro"
        over one domain is not a macro.

    Returns [(macro_mean_pred, macro_emp_freq, lo, hi, total_count), ...].
    """
    domains = sorted(cells)
    P = {d: np.array([r["parsed_prob"] for r in cells[d]], dtype=float) for d in domains}
    Y = {d: np.array([r["label"] for r in cells[d]], dtype=float) for d in domains}

    pooled = np.concatenate([P[d] for d in domains])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    nb = len(edges) - 1
    idx = {d: np.digitize(P[d], edges[1:-1], right=True) for d in domains}

    def macro_bins(sel: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(macro_x, macro_y, n_domains_contributing) per bin for one (re)sample."""
        fx = np.full((len(domains), nb), np.nan)
        fy = np.full((len(domains), nb), np.nan)
        cnt = np.zeros(nb, dtype=int)
        for di, d in enumerate(domains):
            i = idx[d][sel[d]]
            p, y = P[d][sel[d]], Y[d][sel[d]]
            for b in range(nb):
                m = i == b
                c = int(m.sum())
                cnt[b] += c
                if c >= min_count:
                    fx[di, b] = p[m].mean()
                    fy[di, b] = y[m].mean()
        # An all-nan column is expected (a bin no domain occupies densely enough); those bins are
        # dropped below. nanmean warns rather than raising, so silence just that.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(fx, axis=0), np.nanmean(fy, axis=0), (~np.isnan(fy)).sum(axis=0)

    full = {d: np.arange(len(P[d])) for d in domains}
    mx, my, ndom = macro_bins(full)

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, nb), np.nan)
    for b in range(n_boot):
        # Resample episodes within domain -- the same clustering macro_ci uses (each episode is its own
        # cluster here, since a published primary row is one forward pass).
        sel = {d: rng.integers(0, len(P[d]), len(P[d])) for d in domains}
        boot[b] = macro_bins(sel)[1]

    out = []
    for b in range(nb):
        if ndom[b] < min_domains or np.isnan(my[b]):
            continue
        col = boot[:, b]
        col = col[~np.isnan(col)]
        if n_boot <= 0:
            # CI deliberately not computed (a caller that does not shade its bands). NaN, not lo=hi=point,
            # which would read as a zero-width interval instead of an absent one.
            lo = hi = float("nan")
        else:
            lo, hi = (float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))) if col.size else (my[b], my[b])
        out.append((float(mx[b]), float(my[b]), lo, hi, int(cnt_total(idx, domains, b))))
    return out


def cnt_total(idx: dict[str, np.ndarray], domains: list[str], b: int) -> int:
    return int(sum(int((idx[d] == b).sum()) for d in domains))


def curves_for(
    testset_slug: str, keys: list[str], n_bins: int, n_boot: int, root: Path | None = None, macro_bins_n: int = 0
) -> dict[str, dict[str, list[tuple]]]:
    """{roster_key: {domain: [(mean_pred, emp_freq, lo, hi, count), ...]}}."""
    out: dict[str, dict[str, list[tuple]]] = {}
    for key in keys:
        e = ROSTER[key]
        cells = load_cells(testset_slug, e.eval_key, root=root)
        if not cells:
            print(f"[reliability] skip {key}: no cells under {e.eval_key}", file=sys.stderr)
            continue
        per_domain = {}
        for d, rows in cells.items():
            probs = [r["parsed_prob"] for r in rows]
            labels = [r["label"] for r in rows]
            per_domain[d] = reliability_curve_ci(probs, labels, n_bins=n_bins, n_boot=n_boot)
        if macro_bins_n and len(cells) > 1:
            mc = macro_curve(cells, macro_bins_n, n_boot)
            per_domain[MACRO_KEY] = mc
            # Report dropped bins rather than letting the panel imply full coverage: a row that crowds
            # its predictions into a narrow band loses shared bins to per-domain sparsity, and how many
            # it lost is a property of that model worth seeing.
            if len(mc) < macro_bins_n:
                print(
                    f"[reliability] {e.label}: macro panel kept {len(mc)}/{macro_bins_n} bins "
                    f"(rest dropped for per-domain sparsity)",
                    file=sys.stderr,
                )
        out[key] = per_domain
    return out


def render(
    curves: dict[str, dict[str, list[tuple]]],
    out_base: Path,
    *,
    show_ci: bool,
    n_bins: int,
    macro_bins_n: int = 0,
) -> None:
    """One row of square panels: the three domains, then the macro-average LAST.

    A macro panel spanning a full row above the per-domain panels would be taller than the 9.25 in text
    block at `width=\\textwidth`. A single row at `TEXT_WIDTH_IN` fits, and putting the macro panel last reads as "and here is the summary" rather than
    burying the per-domain detail below the fold.

    All model curves are SOLID. `roster_palette` separates same-hue variants with `linestyle`, which is
    unusable here: dashed grey is the perfect-calibration reference on a reliability diagram, so a dashed
    model curve competes with it. `plot_color` does that separation by shade instead, leaving dash
    reserved for the reference line.

    Every panel is square and shares (0,1) limits, so deviation from the diagonal is comparable by eye.
    """
    from matplotlib.lines import Line2D

    domains = [d for d in DOMAINS if any(d in v for v in curves.values())]
    has_macro = any(MACRO_KEY in v for v in curves.values())
    if not domains and not has_macro:
        print("[reliability] nothing to plot", file=sys.stderr)
        return

    def draw(ax, d) -> None:
        ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=GREY["ref"], zorder=1)
        for key, per_domain in curves.items():
            pts = per_domain.get(d)
            if not pts:
                continue
            col = plot_color(ROSTER[key])
            xs = [q[0] for q in pts]
            ys = [q[1] for q in pts]
            ax.plot(xs, ys, "-", color=col, lw=1.4, marker="o", ms=2.2, zorder=3, alpha=0.95)
            if show_ci:
                ax.fill_between(xs, [q[2] for q in pts], [q[3] for q in pts], color=col, alpha=0.13, lw=0, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=FONT["tick"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Macro LAST, per the layout above. Panels are square by construction: `panel_row` divides the width
    # evenly and the row height is set to that same panel width.
    panels = list(domains) + ([MACRO_KEY] if has_macro else [])
    L_, R_, GAP_ = 0.52, 0.10, 0.26
    pw = (TEXT_WIDTH_IN - L_ - R_ - (len(panels) - 1) * GAP_) / len(panels)
    TOP, XLAB, LEG = 0.30, 0.42, 0.46
    fig = new_figure(TOP + pw + XLAB + LEG)
    axes = panel_row(fig, len(panels), y=LEG + XLAB, h=pw, left=L_, right=R_, gap=GAP_)

    for ax, d in zip(axes, panels):
        draw(ax, d)
        ax.set_title("All domains (macro)" if d == MACRO_KEY else dn.domain(d), fontsize=FONT["title"])
        ax.set_xlabel(r"Model $p(\theta)$", fontsize=FONT["label"])
    axes[0].set_ylabel("empirical frequency", fontsize=FONT["label"])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = [
        Line2D([], [], color=plot_color(ROSTER[k]), ls="-", lw=1.6, marker="o", ms=2.6, label=ROSTER[k].label)
        for k in curves
    ]
    handles.append(Line2D([], [], color=GREY["ref"], ls="--", lw=1.1, label="perfect calibration"))
    bin_note = f"{n_bins} equal-mass bins/domain"
    if has_macro:
        bin_note += f"; macro {macro_bins_n} shared bins"
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=int(np.ceil(len(handles) / 2)),
        frameon=False,
        fontsize=FONT["legend"],
        handlelength=1.6,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.010),
    )
    save_figure(fig, out_base, tag="reliability", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None, help="override the eval tree root")
    ap.add_argument("--keys", default=None, help="comma list of roster keys (default: full roster order)")
    ap.add_argument("--out-dir", default="outputs/reliability")
    ap.add_argument("--n-bins", type=int, default=N_BINS, help=f"equal-mass bins per curve (default {N_BINS})")
    ap.add_argument("--n-boot", type=int, required=True, help="bootstrap draws for the per-bin CI")
    ap.add_argument(
        "--macro-bins",
        type=int,
        default=10,
        help="bins for the extra All (macro) panel: per-domain bin frequency, then equal-weight mean over "
        "domains, matching macro_bootstrap. Fewer than --n-bins because the edges are SHARED across "
        "domains, so per-domain occupancy is the binding constraint. 0 disables the panel.",
    )
    ap.add_argument(
        "--ci",
        action="store_true",
        help="shade the per-bin 95%% CI. Off by default: with the full roster overlaid the bands occlude "
        "each other; most useful with --keys narrowed to two or three rows.",
    )
    args = ap.parse_args()

    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else list(ORDER)
    if unknown := [k for k in keys if k not in ROSTER]:
        raise SystemExit(f"unknown roster keys: {unknown} (known: {list(ORDER)})")

    curves = curves_for(slug, keys, args.n_bins, args.n_boot, root=root, macro_bins_n=args.macro_bins)
    if not curves:
        raise SystemExit(f"no results found under {eval_paths.result_dir(slug, '', root=root).parent}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reliability_curves.json").write_text(
        json.dumps(
            {
                ROSTER[k].label: {
                    d: [{"mean_pred": p[0], "emp_freq": p[1], "lo": p[2], "hi": p[3], "count": p[4]} for p in pts]
                    for d, pts in per_domain.items()
                }
                for k, per_domain in curves.items()
            },
            indent=2,
        )
    )
    render(
        curves,
        out_dir / "reliability",
        show_ci=args.ci,
        n_bins=args.n_bins,
        macro_bins_n=args.macro_bins,
    )
    print(f"[reliability] curve data -> {out_dir / 'reliability_curves.json'}")


if __name__ == "__main__":
    main()
