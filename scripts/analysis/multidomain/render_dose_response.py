"""Dose-response curves on the published multi-domain TEST set: metric vs. amount of evidence shown.

Reads the canonical eval tree through `scripts/lib/eval_paths.py`, takes its roster and palette from `scripts/lib/roster_palette.py`, and macro-averages per domain the way everything else in
this directory does.

**This figure requires the `sensitivity` config, and `--tag` is what selects it.** `sensitivity` is the
dense nested sweep — the SAME 256 episodes per domain re-scored at every k=0..16 — so a curve is a
within-episode comparison and its shape is not confounded by which episodes happen to sit at which k.
`primary` cannot produce this figure at all: it carries one k per episode, so a "curve" over it would
compare different episodes at every point. Because both configs share a result directory (the testset slug
keys on repo + revision, not config), the tag is the only thing that distinguishes them — hence the
default is `ctx*` here and the script refuses a `primary` tag rather than silently drawing a 1-point curve.

**Two things it plots that are easy to get wrong.**

*Macro-averaging.* At each k the statistic is computed per domain and then equal-weight averaged, with the
CI from one bootstrap that resamples episodes **within** domain and macro-averages each draw
(`scripts/lib/macro_bootstrap.py`). Pooling all 768 episodes at a k and resampling that would be a micro
average, and would understate the interval.

*Parse-rate survivorship.* Unparsed episodes are excluded from metrics, never scored as wrong — so if the
parse rate falls with k, a curve computed on survivors is confounded by exactly the variable on the x
axis, and the confound points the flattering way (the episodes a model truncates on are the ones it found
hardest). The parse-rate panel is therefore part of this figure rather than a separate check, and
`--min-parse-rate` prints a loud table of every (model, domain, k) below the bar. Read the two together:
a metric improving with k while parse rate falls with k is not yet evidence of anything.

`k=0` and `k=1` sit OUTSIDE the training distribution (`context_size_range = [2, 16]` in
`train/_common/classification-base.toml`), so the trained rows are extrapolating there. That range is
shaded rather than annotated per-curve, because it applies to the figure, not to any one model.
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
from calibration_metrics import compute_calibration  # noqa: E402
from figure_io import FONT, GREY, new_figure, panel_grid, panel_row, save_figure  # noqa: E402
from macro_bootstrap import macro_ci, macro_point  # noqa: E402
from multidomain_cells import DOMAINS, cell_paths_for_tag  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

# Metrics this figure can draw, grouped by HOW they are computed -- the only distinction that matters
# downstream, because it decides whether a metric needs the pairing pre-transform.
#
# POINTWISE: a function of the forecasts AT one k. `ptrue` is the mean probability on the true label; the rest come
# from calibration_metrics so they cannot drift from the master table's definitions.
# PAIRED: needs the SAME episode at a second k, so it cannot be a function of one k's records. These are
# the Bayes factors, which divide out the model's own prior and therefore isolate the in-context update
# from the quality of where the model started.
POINTWISE: tuple[str, ...] = ("ptrue", "brier", "auroc", "smece", "mean_p", "sharpness")
PAIRED: tuple[str, ...] = ("logbf", "logbf_step")
METRICS: tuple[str, ...] = POINTWISE + PAIRED
DEFAULT_METRICS: tuple[str, ...] = ("ptrue", "brier", "auroc")

# What each paired metric measures against: the model's own prior, or the previous k.
AGAINST: dict[str, str] = {"logbf": "prior", "logbf_step": "previous"}

PRETTY = {
    "ptrue": r"mean $p(\theta = \theta^*)$",  # lowercase p: the paper's notation
    "logbf": "cumulative log BF vs. own prior (db)",
    "logbf_step": "marginal log BF of the k-th stub (db)",
    "mean_p": r"mean $p(\theta)$",
    "sharpness": "sharpness (var of forecasts)",
}
# Higher-is-better for these; everything else in METRICS comes from display_names.
_HIGHER = {"ptrue", "auroc", "sharpness", "logbf", "logbf_step"}

# Shared with render_woe_structure so the two are on one scale.
EPS = 1e-3
DECIBAN = 10.0 / np.log(10.0)


def metric_label(name: str) -> str:
    return PRETTY.get(name, dn.metric(name))


def _statistic(name: str):
    """`list[record] -> float` for one metric, over one domain's episodes at one k.

    Returns NaN rather than raising on a degenerate input (empty after parse filtering, or single-class,
    which makes AUROC undefined). A NaN is dropped from that draw's macro-average by `macro_point`,
    which is the right behaviour for a bootstrap draw that happened to lose a domain.
    """

    def f(rows: list[dict]) -> float:
        if not rows:
            return float("nan")
        p = np.clip(np.array([r["parsed_prob"] for r in rows], dtype=float), 0.0, 1.0)
        y = np.array([r["label"] for r in rows], dtype=float)
        if name == "ptrue":
            return float(np.mean(np.where(y == 1, p, 1.0 - p)))
        if name == "mean_p":
            return float(np.mean(p))
        if name == "brier":
            return float(np.mean((p - y) ** 2))
        if name == "sharpness":
            return float(np.var(p))
        if len(np.unique(y)) < 2 and name == "auroc":
            return float("nan")  # AUROC undefined on a single-class draw
        rep = compute_calibration(p, y, with_smece=(name == "smece"))
        v = getattr(rep, name, None)
        return float("nan") if v is None else float(v)

    return f


def _logit(p: float) -> float:
    return float(np.log(min(max(float(p), EPS), 1.0 - EPS) / (1.0 - min(max(float(p), EPS), 1.0 - EPS))))


def paired_logbf(by_k: dict, ks: list[int], *, against: str) -> dict[int, list[dict]]:
    """{k: [{domain, episode_id, value}]} — log Bayes factor toward the GROUND TRUTH, in decibans.

        value = s * [logit(p_k) - logit(p_ref)] * 10/ln 10,   s = +1 if the episode is supported else -1

    **Why this exists.** Every POINTWISE metric is a level, so a model that simply starts closer to the
    truth scores better at every k whether or not it responds to evidence at all. A Bayes factor is a
    RATIO against that model's own earlier forecast, so the prior cancels exactly and what remains is only
    the update the context produced. `s` orients it toward the ground truth: positive means the evidence
    moved the model the right way, where an unsigned version would reward confident movement either way.

    `against` picks the reference:
      "prior"    -- ks[0] (k=0 in practice): CUMULATIVE, "how far has the context moved this model".
      "previous" -- the preceding k in `ks`: MARGINAL, "what did the k-th stub alone buy". Same data
                    differentiated, but only the marginal view makes saturation legible; a model that
                    stops responding looks merely plateaued in the cumulative one.

    Paired per episode. An episode with no reference forecast is DROPPED rather than compared against some
    other episode's, and "previous" uses the preceding k PRESENT in `ks` rather than k-1 arithmetically,
    so a coverage gap compares adjacent available points instead of crediting one stub with two stubs'
    movement. The first k has no predecessor, so "previous" starts one point in.

    This is the same quantity `render_woe_structure` calls a per-stub weight of evidence, differing in one
    deliberate way: signed toward the ground truth rather than by the model's own direction.
    """
    if against not in ("prior", "previous"):
        raise ValueError(f"against must be 'prior' or 'previous', got {against!r}")

    def indexed(k: int) -> dict[tuple[str, str], dict]:
        return {(r["domain"], str(r["episode_id"])): r for r in by_k.get(k, [])}

    out: dict[int, list[dict]] = {}
    ref = indexed(ks[0]) if ks else {}
    for i, k in enumerate(ks):
        cur = indexed(k)
        if against == "previous":
            if i == 0:
                ref = cur  # no predecessor; this k contributes no point
                continue
        rows = [
            {
                "domain": r["domain"],
                "episode_id": r["episode_id"],
                "value": (1.0 if int(r["label"]) == 1 else -1.0)
                * (_logit(r["parsed_prob"]) - _logit(ref[key]["parsed_prob"]))
                * DECIBAN,
            }
            for key, r in cur.items()
            if key in ref
        ]
        out[k] = rows
        if against == "previous":
            ref = cur
    return out


def _mean_value(rows: list[dict]) -> float:
    """Mean of the pre-transformed `value` field -- for a log BF, the GEOMETRIC mean Bayes factor.

    Not the arithmetic mean of BF: it is multiplicative and heavy-tailed (|WoE| saturates near 60 db, so
    BF reaches ~1e6), and E[BF] would be set by a handful of episodes rather than by typical behaviour.
    """
    return float(np.mean([r["value"] for r in rows])) if rows else float("nan")


def ks_present(slug: str, eval_key: str, tag: str, root: Path | None) -> list[int]:
    """The k values this model has cells for in EVERY domain, ascending.

    Intersected across domains rather than unioned: a curve point that silently covers two domains at one
    k and three at the next is not a macro-average of a fixed domain set, and the drop would read as a
    change in the model rather than in coverage.
    """
    per: list[set[int]] = []
    for d in DOMAINS:
        got = set()
        for p in cell_paths_for_tag(slug, eval_key, d, tag, root=root):
            got.add(int(p.stem.rsplit("_ctx", 1)[1]))
        per.append(got)
    return sorted(set.intersection(*per)) if per and all(per) else []


def load_by_k(slug: str, eval_key: str, tag: str, root: Path | None) -> tuple[dict, dict]:
    """({k: [parsed record + domain]}, {(domain, k): (n_attempted, n_parsed)}) for one model."""
    by_k: dict[int, list[dict]] = {}
    attempts: dict[tuple[str, int], tuple[int, int]] = {}
    for d in DOMAINS:
        for path in cell_paths_for_tag(slug, eval_key, d, tag, root=root):
            k = int(path.stem.rsplit("_ctx", 1)[1])
            rows = json.loads(path.read_text())["results"]
            attempts[(d, k)] = (len(rows), sum(r.get("parsed_prob") is not None for r in rows))
            for r in rows:
                if r.get("parsed_prob") is not None:
                    by_k.setdefault(k, []).append({**r, "domain": d})
    return by_k, attempts


def _rows_and_stat(by_k: dict, ks: list[int], metric: str):
    """({k: rows}, `list[record] -> float`) for `metric` — the one place the two kinds diverge.

    A PAIRED metric is pre-transformed into per-episode scalars here; after this line it is an ordinary
    mean and every downstream step (the within-domain bootstrap, the caching, both figure layouts, the
    parse-rate check) is shared with the POINTWISE metrics unchanged.
    """
    if metric in PAIRED:
        return paired_logbf(by_k, ks, against=AGAINST[metric]), _mean_value
    return {k: by_k.get(k, []) for k in ks}, _statistic(metric)


def curve(by_k: dict, ks: list[int], metric: str, n_boot: int, seed: int) -> list[dict]:
    """[{k, point, lo, hi, n}] — macro-averaged at each k, CI resampling episodes within domain."""
    rows_by_k, stat = _rows_and_stat(by_k, ks, metric)
    out = []
    for k in ks:
        rows = rows_by_k.get(k, [])
        if n_boot > 0:
            pt, lo, hi = macro_ci(rows, stat, n_boot=n_boot, seed=seed, stat_id=f"dose:{metric}")
        else:
            pt, lo, hi = macro_point(rows, stat), float("nan"), float("nan")
        out.append({"k": k, "point": pt, "lo": lo, "hi": hi, "n": len(rows)})
    return out


def parse_curve(attempts: dict, ks: list[int]) -> list[dict]:
    """Macro parse rate per k, plus the per-domain rates that the macro average hides.

    The macro number is what goes on the panel; the per-domain values go in the JSON because the skew is
    known to be by domain (on `primary`, thinking-RL lost 4.0% of tahoe but 0.2% of depmap), and a single
    averaged line would hide the one domain doing the dropping.
    """
    out = []
    for k in ks:
        per = {}
        for d in DOMAINS:
            att, ok = attempts.get((d, k), (0, 0))
            if att:
                per[d] = ok / att
        out.append(
            {
                "k": k,
                "macro": float(np.mean(list(per.values()))) if per else float("nan"),
                "per_domain": per,
                "attempted": {d: attempts.get((d, k), (0, 0))[0] for d in DOMAINS},
                "parsed": {d: attempts.get((d, k), (0, 0))[1] for d in DOMAINS},
            }
        )
    return out


def domain_curve(by_k: dict, ks: list[int], metric: str, domain: str, n_boot: int, seed: int) -> list[dict]:
    """[{k, point, lo, hi, n}] for ONE domain — the per-domain counterpart of `curve`.

    Still routed through `macro_ci` even though there is nothing to macro-average over: restricted to one
    domain it degenerates to a plain episode bootstrap of that domain, which is exactly the right
    interval, and reusing it keeps the resampling scheme identical to every other figure rather than
    hand-rolling a second one that could drift.
    """
    rows_by_k, stat = _rows_and_stat(by_k, ks, metric)
    out = []
    for k in ks:
        rows = [r for r in rows_by_k.get(k, []) if r["domain"] == domain]
        if n_boot > 0 and rows:
            pt, lo, hi = macro_ci(rows, stat, n_boot=n_boot, seed=seed, stat_id=f"dose:{metric}:{domain}")
        else:
            pt, lo, hi = (macro_point(rows, stat) if rows else float("nan")), float("nan"), float("nan")
        out.append({"k": k, "point": pt, "lo": lo, "hi": hi, "n": len(rows)})
    return out


def render_by_domain(
    series: dict[str, dict],
    metric: str,
    out_base: Path,
    train_k: tuple[int, int] | None,
) -> None:
    """ONE statistic, four panels: the three domains then the macro-average LAST.

    The companion to `render` (which holds the domain set fixed at the macro-average and varies the
    metric across panels). This holds the metric fixed and varies the domain, which is the view that
    answers "does the evidence response look the same everywhere, or is the macro-average hiding a domain
    that behaves differently" — a question the macro-only figure cannot address at all.

    Macro last and y shared across panels, matching `render_reliability` / `render_roc`: one statistic on
    one scale is the whole point, and per-panel autoscaling would make a flat domain look responsive.
    """
    from matplotlib.lines import Line2D

    panels = list(DOMAINS) + ["__macro__"]
    L_, R_, GAP_ = 0.58, 0.10, 0.30
    PH, TOP, BOT, CAP, LEG = 1.34, 0.48, 0.46, 0.18, 0.46
    figh = TOP + PH + BOT + CAP + LEG
    fig = new_figure(figh)
    axes = panel_row(fig, len(panels), y=LEG + CAP + BOT, h=PH, left=L_, right=R_, gap=GAP_)

    kmax = max(max(d["k"] for d in s["parse"]) for s in series.values())
    vals = [
        v
        for s in series.values()
        for p in panels
        for d in (s["curves"][metric] if p == "__macro__" else s["by_domain"][metric][p])
        for v in (d["point"], d["lo"], d["hi"])
        if not np.isnan(v)
    ]
    ylo, yhi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    pad = 0.06 * max(yhi - ylo, 1e-9)

    for ax, p in zip(axes, panels):
        if train_k:
            # Shade the region the trained rows never saw, so extrapolation is visible rather than implied.
            ax.axvspan(-0.4, train_k[0] - 0.5, color=GREY["fill"], alpha=0.55, lw=0, zorder=0)
        for key, s in series.items():
            pts = s["curves"][metric] if p == "__macro__" else s["by_domain"][metric][p]
            xs = [d["k"] for d in pts]
            ys = np.array([d["point"] for d in pts], dtype=float)
            col = plot_color(ROSTER[key])
            ax.plot(xs, ys, color=col, lw=1.4, marker="o", ms=2.0, zorder=3)
            lo = np.array([d["lo"] for d in pts], dtype=float)
            hi = np.array([d["hi"] for d in pts], dtype=float)
            if np.isfinite(lo).any():
                ax.fill_between(xs, lo, hi, color=col, alpha=0.14, lw=0, zorder=2)
        ax.set_title("All domains (macro)" if p == "__macro__" else dn.domain(p), fontsize=FONT["title"])
        ax.set_xlabel("evidence stubs shown (k)", fontsize=FONT["label"])
        ax.set_xlim(-0.5, kmax + 0.5)
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.tick_params(labelsize=FONT["tick"])
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    arrow = "↑" if metric in _HIGHER else "↓"
    axes[0].set_ylabel(f"{metric_label(metric)} ({arrow})", fontsize=FONT["label"])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    handles = [
        Line2D([], [], color=plot_color(ROSTER[k]), lw=3.6, solid_capstyle="butt", label=ROSTER[k].label)
        for k in series
    ]
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
    save_figure(fig, out_base, tag=f"dose:by-domain:{metric}", tight=False)


def render(
    series: dict[str, dict],
    metrics: list[str],
    out_base: Path,
    train_k: tuple[int, int] | None,
    show_parse: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    panels = list(metrics) + (["parse_rate"] if show_parse else [])
    # 2 columns up to 4 panels (matches render_prob_hist): at 4 panels a 3-wide grid leaves a row with
    # one plot and two blank cells, which pushes the shared legend far below the figure.
    ncol = 2 if len(panels) <= 4 else 3
    nrow = int(np.ceil(len(panels) / ncol))
    # Absolute inches at TEXT_WIDTH_IN, so LaTeX places the figure at `width=\textwidth` without rescaling
    # and the declared font sizes are the printed ones.
    L_, R_, GAP_, RGAP = 0.60, 0.12, 0.52, 0.52
    PH, TOP, BOT, LEG = 1.55, 0.58, 0.42, 0.46
    figh = TOP + nrow * PH + (nrow - 1) * RGAP + BOT + LEG
    fig = new_figure(figh)
    flat = panel_grid(fig, nrow, ncol, y=LEG + BOT, h=PH, row_gap=RGAP, left=L_, right=R_, gap=GAP_)

    for ax, name in zip(flat, panels):
        for key, s in series.items():
            e = ROSTER[key]
            col = plot_color(e)
            pts = s["parse"] if name == "parse_rate" else s["curves"][name]
            xs = [d["k"] for d in pts]
            ys = np.array([d["macro"] if name == "parse_rate" else d["point"] for d in pts], dtype=float)
            # SOLID for every row. `roster_palette` declares a dashed linestyle for the same-hue variant
            # (base-thinking) and for the two frontier rows, but `plot_color` already separates the shared
            # hue by shade, so dashing it too double-encodes one distinction — and the frontier rows have
            # unique colours, so their dash encodes nothing at all. Same convention as render_reliability.
            ax.plot(xs, ys, color=col, lw=1.5, marker="o", ms=2.4, label=e.label)
            if name != "parse_rate":
                lo = np.array([d["lo"] for d in pts], dtype=float)
                hi = np.array([d["hi"] for d in pts], dtype=float)
                if np.isfinite(lo).any():
                    ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        # Shade the region the trained rows never saw, so extrapolation is visible rather than implied.
        if train_k:
            ax.axvspan(-0.4, train_k[0] - 0.5, color=GREY["fill"], alpha=0.55, lw=0, zorder=0)
        if name == "parse_rate":
            ax.set_ylabel("parse rate (macro over domains)", fontsize=FONT["label"])
            ax.set_title("Parse rate — survivorship check", fontsize=FONT["title"])
            ax.set_ylim(-0.02, 1.02)
            ax.axhline(1.0, ls=":", lw=1, color=GREY["ref"], zorder=0)
        else:
            arrow = "↑" if name in _HIGHER else "↓"
            ax.set_ylabel(f"{metric_label(name)} ({arrow} better)", fontsize=FONT["label"])
            ax.set_title(metric_label(name), fontsize=FONT["title"])
        ax.set_xlabel("evidence stubs shown (k)", fontsize=FONT["label"])
        ax.tick_params(labelsize=FONT["tick"])
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.5, max(max(d["k"] for d in s["parse"]) for s in series.values()) + 0.5)
    for ax in flat[len(panels) :]:
        ax.set_visible(False)

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=int(np.ceil(len(labels) / 2)),
        frameon=False,
        fontsize=FONT["legend"],
        handlelength=1.6,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.010),
    )
    save_figure(fig, out_base, tag="dose", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/sensitivity/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None, help="override the eval tree root")
    ap.add_argument(
        "--tag",
        default="ctx*",
        help="cells to read; 'ctx*' (default) is every k of the sensitivity sweep. A 'primary' tag is "
        "refused: that config holds one k per episode and cannot express a dose-response curve.",
    )
    ap.add_argument("--keys", default=",".join(ORDER), help="comma list of roster keys (default: full roster)")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help=f"panels, from {list(METRICS)}")
    ap.add_argument(
        "--by-domain-metric",
        default="ptrue",
        help=f"also draw dose_response_by_domain_<metric>: ONE statistic, one panel per domain plus the "
        f"macro-average. From {list(METRICS)}; empty string disables it. Each (metric, domain) pair is its "
        "own bootstrap, so this is priced per metric — hence one at a time rather than all of them.",
    )
    ap.add_argument("--n-boot", type=int, required=True, help="bootstrap draws per point; 0 = points only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/dose_response")
    ap.add_argument(
        "--train-k-range",
        default="2,16",
        help="training context range to shade outside of (train/_common/classification-base.toml's "
        "context_size_range); empty string disables the shading",
    )
    ap.add_argument("--no-parse-panel", action="store_true", help="drop the parse-rate panel (not advised)")
    ap.add_argument(
        "--min-parse-rate",
        type=float,
        default=0.98,
        help="print every (model, domain, k) whose parse rate falls below this",
    )
    args = ap.parse_args()

    if args.tag == "primary":
        raise SystemExit(
            "--tag primary cannot produce a dose-response curve: the primary config carries ONE k per "
            "episode, so each k would be a different episode set. Use the sensitivity sweep (--tag 'ctx*')."
        )
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if unknown := [m for m in metrics if m not in METRICS]:
        raise SystemExit(f"unknown metrics {unknown}; choose from {list(METRICS)}")
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if unknown := [k for k in keys if k not in ROSTER]:
        raise SystemExit(f"unknown roster keys {unknown}; known: {list(ROSTER)}")
    train_k = None
    if args.train_k_range.strip():
        a, b = (int(x) for x in args.train_k_range.split(","))
        train_k = (a, b)

    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None

    # The by-domain figure's 4th panel is the macro-average, which is read from `curves` -- so its metric
    # has to be computed there too even if the caller did not list it in --metrics.
    if args.by_domain_metric and args.by_domain_metric not in metrics:
        metrics = list(metrics) + [args.by_domain_metric]

    series: dict[str, dict] = {}
    for key in keys:
        ek = ROSTER[key].eval_key
        ks = ks_present(slug, ek, args.tag, root)
        if not ks:
            print(f"[dose] skip {key}: no '{args.tag}' cells in all {len(DOMAINS)} domains", file=sys.stderr)
            continue
        by_k, attempts = load_by_k(slug, ek, args.tag, root)
        series[key] = {
            "label": ROSTER[key].label,
            "eval_key": ek,
            "ks": ks,
            "curves": {m: curve(by_k, ks, m, args.n_boot, args.seed) for m in metrics},
            # Per-domain curves for the by-domain figure. Only the requested statistic, because each
            # (metric, domain) pair is its own bootstrap and computing all six across three domains
            # would quadruple the run for panels nothing draws.
            "by_domain": {
                m: {d: domain_curve(by_k, ks, m, d, args.n_boot, args.seed) for d in DOMAINS}
                for m in ([args.by_domain_metric] if args.by_domain_metric else [])
            },
            "parse": parse_curve(attempts, ks),
        }
        print(f"[dose] {ROSTER[key].label:24s} k={ks[0]}..{ks[-1]} ({len(ks)} points)")
    if not series:
        raise SystemExit(f"no '{args.tag}' cells under {eval_paths.result_dir(slug, '', root=root).parent}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dose_response.json").write_text(
        json.dumps({"testset": args.testset, "revision": args.revision, "tag": args.tag, "rows": series}, indent=2)
    )

    # Survivorship report. Printed even when everything passes, so a clean run says so explicitly
    # rather than being silent in a way that looks the same as never having checked.
    bad = [
        (s["label"], d, p["k"], r)
        for s in series.values()
        for p in s["parse"]
        for d, r in p["per_domain"].items()
        if r < args.min_parse_rate
    ]
    print(f"\n[dose] parse-rate check (<{args.min_parse_rate:.0%}): {len(bad)} (model, domain, k) cells")
    for lbl, d, k, r in sorted(bad, key=lambda x: x[3])[:40]:
        print(f"   {lbl:24s} {d:7s} k={k:<3d} parse={r:.1%}")
    if bad:
        print("   ^ curves at these k are computed on survivors; see the parse-rate panel.")

    # BY METRIC: domain set fixed at the macro-average, metric varies across panels.
    render(
        series,
        metrics,
        out_dir / "dose_response_by_metric",
        train_k,
        not args.no_parse_panel,
    )
    # BY DOMAIN: metric fixed, domain varies. Answers whether the macro-average hides a domain that
    # responds to evidence differently -- which the by-metric figure cannot show at all.
    if args.by_domain_metric:
        render_by_domain(
            series,
            args.by_domain_metric,
            out_dir / f"dose_response_by_domain_{args.by_domain_metric}",
            train_k,
        )
    print(f"[dose] data -> {out_dir / 'dose_response.json'}")


if __name__ == "__main__":
    main()
