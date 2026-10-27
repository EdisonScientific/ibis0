"""Score the roster on the primary test set with and without the train/test-overlapping episodes.

Reads the episode ids that `train_test_overlap.py` flagged and, for every roster row, computes Brier,
AUROC, ECE and sMECE three ways: on the full primary cells, on the overlapping episodes alone, and on the
remainder ("filtered"). Emits the numbers as CSV/JSON and the paper's figure (Appendix A.7): a grid of
metrics by domains (plus the equal-weight macro), with a full and a filtered bar per model and the
filtered-minus-full delta printed above a pair when it reaches `--delta-label-min`.

Why the untrained and frontier rows are in the figure at all: they never saw the training pools, so
their full-vs-filtered movement measures how hard the overlapping subset is on its own. A trained row
that moves by the same amount is showing a selection effect, not memorization. Read the figure as a
comparison across rows, not as a per-row correction.

Macro is the equal-weight mean of the three per-domain values, as everywhere else in this directory, so
it does not re-weight the domains when filtering leaves them at different sizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import display_names as dn  # noqa: E402
import eval_paths  # noqa: E402
from calibration_metrics import compute_calibration  # noqa: E402
from figure_io import FONT, GREY, new_figure, panel_grid, save_figure  # noqa: E402
from multidomain_cells import DOMAINS, load_cells  # noqa: E402
from roster_palette import ORDER, ROSTER  # noqa: E402

METRICS = ("brier", "auroc", "smece")
SUBSETS = ("full", "leaked", "filtered")
# Condition colours, not model colours: the bars are grouped by model and the colour says which test set.
# Solid mid-blue for the full set; a light, hatched blue for the filtered set so the pair reads in greyscale.
COLORS = {"full": "#2171b5", "filtered": "#c6dbef"}
HATCH = "////"
LABELS = {"full": "Full dataset", "filtered": "Filtered dataset"}


def subset_metrics(records: list[dict], leaked: set[str]) -> dict[str, dict]:
    """{subset: {n, brier, auroc, ece, smece, base_rate}} for one (model, domain) cell.

    A subset with fewer than two episodes or a single label class has no defined AUROC and gets `n` only.
    """
    ids = np.array([r["episode_id"] for r in records])
    p = np.array([r["parsed_prob"] for r in records], dtype=float)
    y = np.array([r["label"] for r in records], dtype=float)
    is_leaked = np.array([i in leaked for i in ids], dtype=bool)
    masks = {"full": np.ones(len(p), bool), "leaked": is_leaked, "filtered": ~is_leaked}
    out = {}
    for name, m in masks.items():
        n = int(m.sum())
        if n < 2 or len(set(y[m])) < 2:
            out[name] = {"n": n}
            continue
        c = compute_calibration(p[m], y[m], n_bins=10, with_smece=True)
        out[name] = {
            "n": n,
            "base_rate": c.base_rate,
            "brier": c.brier,
            "auroc": c.auroc,
            "ece": c.ece,
            "smece": c.smece,
        }
    return out


def macro(per_domain: dict[str, dict], subset: str, metric: str) -> float | None:
    """Equal-weight mean over domains; None if any domain lacks the value."""
    vals = [per_domain[d].get(subset, {}).get(metric) for d in DOMAINS if d in per_domain]
    if len(vals) < len(DOMAINS) or any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def collect(slug: str, leaked_ids: dict[str, set[str]], keys: list[str], root: Path | None, tag: str) -> dict:
    """{roster_key: {domain: {subset: metrics}, "macro": {subset: {metric: value}}}}."""
    out = {}
    for key in keys:
        e = ROSTER[key]
        cells = load_cells(slug, e.eval_key, root=root, tag=tag)
        if not cells:
            print(f"[overlap] {e.label}: no cells under {slug}/{e.eval_key}; row omitted", file=sys.stderr)
            continue
        per_domain = {d: subset_metrics(recs, leaked_ids.get(d, set())) for d, recs in cells.items()}
        per_domain["macro"] = {
            s: {m: macro(per_domain, s, m) for m in ("brier", "auroc", "ece", "smece")} for s in SUBSETS
        }
        out[key] = per_domain
    return out


def flatten(results: dict) -> list[dict]:
    rows = []
    for key, per_domain in results.items():
        for d, subsets in per_domain.items():
            for s, vals in subsets.items():
                rows.append({"model": key, "label": ROSTER[key].label, "domain": d, "subset": s, **vals})
    return rows


# ---------------------------------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------------------------------
def render(results: dict, out_base: Path, delta_min: float) -> None:
    """Metrics as rows, domains (plus macro) as columns; one full and one filtered bar per roster row."""
    keys = [k for k in ORDER if k in results]
    cols = [*DOMAINS, "macro"]
    # Rotated full-name tick labels sit under the bottom row only; the rows above share the x axis.
    panel_h, row_gap, bottom, top = 1.45, 0.20, 1.15, 0.55
    height = bottom + len(METRICS) * panel_h + (len(METRICS) - 1) * row_gap + top
    fig = new_figure(height)
    axes = panel_grid(
        fig, len(METRICS), len(cols), y=bottom, h=panel_h, row_gap=row_gap, left=0.62, right=0.06, gap=0.26
    )
    x = np.arange(len(keys))
    w = 0.38
    for i, metric in enumerate(METRICS):
        # One y range per metric row, so a bar's height means the same thing in every domain column.
        row_vals = [results[k][d].get(sub, {}).get(metric) for k in keys for d in cols for sub in ("full", "filtered")]
        row_top = max([v for v in row_vals if v is not None] or [1.0])
        for j, d in enumerate(cols):
            ax = axes[i * len(cols) + j]
            full = [results[k][d].get("full", {}).get(metric) for k in keys]
            filt = [results[k][d].get("filtered", {}).get(metric) for k in keys]
            full_v = [np.nan if v is None else v for v in full]
            filt_v = [np.nan if v is None else v for v in filt]
            ax.bar(x - w / 2 - 0.02, full_v, w, color=COLORS["full"], label=LABELS["full"], zorder=3)
            ax.bar(
                x + w / 2 + 0.02,
                filt_v,
                w,
                color=COLORS["filtered"],
                hatch=HATCH,
                edgecolor=COLORS["full"],
                linewidth=0.0,
                label=LABELS["filtered"],
                zorder=3,
            )
            top_v = row_top
            for xi, (a, b) in enumerate(zip(full_v, filt_v)):
                if np.isfinite(a) and np.isfinite(b) and abs(b - a) >= delta_min:
                    # Vertical so seven labels fit over seven pairs; horizontal ones collide at this panel width.
                    ax.text(
                        xi,
                        max(a, b) + top_v * 0.03,
                        f"{b - a:+.3f}",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=FONT["annot"],
                        color=GREY["note"],
                    )
            ax.set_ylim(0, top_v * 1.42)
            # Headroom is for the delta labels, not data: keep the ticks inside the metric's own range.
            ceiling = 1.0 if metric == "auroc" else top_v * 1.05
            ax.set_yticks(
                [t for t in MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0, ceiling) if t <= ceiling]
            )
            ax.yaxis.grid(True, color=GREY["rule"], linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(GREY["rule"])
            ax.tick_params(axis="both", length=0, labelsize=FONT["tick"], colors=GREY["ink"])
            ax.set_xticks(x)
            if i == len(METRICS) - 1:
                ax.set_xticklabels(
                    [ROSTER[k].label for k in keys],
                    fontsize=FONT["tick"],
                    rotation=60,
                    ha="right",
                    rotation_mode="anchor",
                )
            else:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(
                    "Macro" if d == "macro" else dn.domain(d),
                    fontsize=FONT["title"],
                    color=GREY["ink"],
                    pad=5,
                )
            if j == 0:
                ax.set_ylabel(
                    f"{dn.metric(metric)} ({dn.direction_arrow(metric)})",
                    fontsize=FONT["label"],
                    color=GREY["ink"],
                    labelpad=6,
                )
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995), fontsize=FONT["legend"])
    save_figure(fig, out_base, tag="overlap")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlap-ids", required=True, help="overlap_ids.json from train_test_overlap.py")
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None, help="override the eval tree root")
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--keys", default=None, help="comma-separated roster subset (default: full roster)")
    ap.add_argument(
        "--delta-label-min", type=float, default=0.005, help="print filtered − full above a pair when |Δ| reaches this"
    )
    ap.add_argument("--out-dir", default="outputs/train_test_overlap")
    args = ap.parse_args()

    spec = json.loads(Path(args.overlap_ids).read_text())
    leaked_ids = {d: set(v["leaked"]) for d, v in spec["domains"].items()}
    leaked_n = {d: len(v) for d, v in leaked_ids.items()}
    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    keys = args.keys.split(",") if args.keys else ORDER

    results = collect(slug, leaked_ids, keys, root, args.tag)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = flatten(results)
    fields = ["model", "label", "domain", "subset", "n", "base_rate", "brier", "auroc", "ece", "smece"]
    with open(out / "overlap_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "overlap_metrics.json").write_text(
        json.dumps(
            {
                "testset": slug,
                "dataset_revision": spec.get("dataset_revision"),
                "rendered_k": spec.get("rendered_k"),
                "leaked_n": leaked_n,
                "results": results,
            },
            indent=1,
        )
    )
    for key in keys:
        if key not in results:
            continue
        m = results[key]["macro"]
        fb, xb = m["full"].get("brier"), m["filtered"].get("brier")
        fa, xa = m["full"].get("auroc"), m["filtered"].get("auroc")
        if None not in (fb, xb, fa, xa):
            print(
                f"[overlap] {ROSTER[key].label:22s} macro Brier {fb:.3f} → {xb:.3f} ({xb - fb:+.3f})   AUROC {fa:.3f} → {xa:.3f} ({xa - fa:+.3f})"
            )
    render(results, out / "train_test_overlap", args.delta_label_min)


if __name__ == "__main__":
    main()
