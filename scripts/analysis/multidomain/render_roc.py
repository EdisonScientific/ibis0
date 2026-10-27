"""ROC curves for the roster on the published multi-domain TEST set.

The discrimination counterpart of `render_reliability.py`: one row of panels (the three domains, then the
macro-average), every roster row overlaid, dashed diagonal = chance. Brier confounds calibration
and discrimination, and the master table separates them only as scalars; this figure shows *where* in the
ranking a model earns or loses its AUROC — whether it separates the easy cases and fails on the rest, or
degrades smoothly. A model can be badly calibrated and still rank perfectly, which is exactly the case the
reliability diagram cannot distinguish and this one can.

**AUROC numbers come from `calibration_metrics.compute_calibration`, not from the plotted curve.** That is
the same tie-corrected Mann-Whitney statistic the master table and the dose-response figure report, so a
number here can never drift from the same number there. `sklearn.roc_curve` is used only for the curve's
geometry.

**The macro curve is vertically averaged; the macro AUROC is the mean of the per-domain AUROCs.** These are
two different objects and the distinction is deliberate. Averaging TPR across domains on a shared FPR grid
is the only way to draw one curve for three domains, but the area under *that* curve is not identical to
the mean of the three areas. The reported macro AUROC is the mean of per-domain AUROCs, because that is
what `macro_bootstrap` defines a macro-average to be and what every other figure in this directory reports
— so the printed number is authoritative and the curve is the shape that number summarises. They agree
closely in practice; where they do not, trust the number.

Each domain contributes equally to the macro curve regardless of how many of its episodes parsed, matching
the equal-weight-per-domain convention. CIs resample episodes **within** domain and re-average, so the band
is a macro CI rather than a pooled one.

Defaults to `--tag primary` (one final forecast per episode — the headline discrimination comparison).
`--tag 'ctx*'` pools the sensitivity sweep, which answers a different question: ranking quality over all
deliberation states at once, where an episode appears 17 times at different evidence levels.
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
from figure_io import FONT, GREY, TEXT_WIDTH_IN, new_figure, panel_row, save_figure  # noqa: E402
from macro_bootstrap import macro_ci  # noqa: E402
from multidomain_cells import DOMAINS, load_cells  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

MACRO_KEY = "__macro__"
# Shared FPR grid the per-domain curves are interpolated onto before averaging. 201 points is fine enough
# that the step structure of a small domain still reads, without pretending to more resolution.
N_GRID = 201


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(np.array([r["parsed_prob"] for r in rows], dtype=float), 0.0, 1.0),
        np.array([r["label"] for r in rows], dtype=int),
    )


def auroc_of(rows: list[dict]) -> float:
    """AUROC via the shared metric module, so this figure agrees with the master table by construction."""
    if not rows:
        return float("nan")
    p, y = _arrays(rows)
    if len(np.unique(y)) < 2:
        return float("nan")  # undefined on a single-class set
    v = compute_calibration(p, y, with_smece=False).auroc
    return float("nan") if v is None else float(v)


def roc_on_grid(rows: list[dict], grid: np.ndarray) -> np.ndarray:
    """TPR interpolated onto `grid` (a shared FPR axis). NaN where AUROC is undefined.

    Interpolation is required for averaging and for a CI at fixed x: each domain's raw ROC has its own
    breakpoints (one per distinct predicted probability), so the curves are not sampled at shared FPR.
    """
    if not rows:
        return np.full_like(grid, np.nan, dtype=float)
    p, y = _arrays(rows)
    if len(np.unique(y)) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, p)
    return np.interp(grid, fpr, tpr)


def curves_for(cells: dict[str, list[dict]], grid: np.ndarray, n_boot: int, seed: int) -> dict:
    """{domain|MACRO: {"tpr", "lo", "hi", "auroc"}} for one model.

    The macro entry averages per-domain TPR at each grid point (equal weight per domain) and reports the
    mean of per-domain AUROCs as its scalar -- see the module docstring on why those are separate.
    """
    out: dict[str, dict] = {}
    per_tpr, per_auc = {}, {}
    for d, rows in cells.items():
        per_tpr[d] = roc_on_grid(rows, grid)
        per_auc[d] = auroc_of(rows)

    rng = np.random.default_rng(seed)
    for d, rows in cells.items():
        band = np.full((max(n_boot, 1), len(grid)), np.nan)
        if n_boot:
            n = len(rows)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                band[b] = roc_on_grid([rows[i] for i in idx], grid)
        out[d] = {
            "tpr": per_tpr[d],
            "lo": np.nanpercentile(band, 2.5, axis=0) if n_boot else np.full(len(grid), np.nan),
            "hi": np.nanpercentile(band, 97.5, axis=0) if n_boot else np.full(len(grid), np.nan),
            "auroc": per_auc[d],
        }

    if len(cells) > 1:
        stack = np.vstack([per_tpr[d] for d in cells])
        macro_tpr = np.nanmean(stack, axis=0)
        band = np.full((max(n_boot, 1), len(grid)), np.nan)
        if n_boot:
            rng2 = np.random.default_rng(seed + 1)
            for b in range(n_boot):
                # Resample WITHIN each domain, then re-average: a macro CI, not a pooled one.
                draws = []
                for d, rows in cells.items():
                    n = len(rows)
                    idx = rng2.integers(0, n, n)
                    draws.append(roc_on_grid([rows[i] for i in idx], grid))
                band[b] = np.nanmean(np.vstack(draws), axis=0)
        aucs = [per_auc[d] for d in cells if np.isfinite(per_auc[d])]
        out[MACRO_KEY] = {
            "tpr": macro_tpr,
            "lo": np.nanpercentile(band, 2.5, axis=0) if n_boot else np.full(len(grid), np.nan),
            "hi": np.nanpercentile(band, 97.5, axis=0) if n_boot else np.full(len(grid), np.nan),
            "auroc": float(np.mean(aucs)) if aucs else float("nan"),
        }
    return out


def render(
    curves: dict[str, dict],
    grid: np.ndarray,
    out_base: Path,
    *,
    show_ci: bool,
    tag: str,
) -> None:
    """One row of square panels: the three domains, then the macro-average LAST.

    Mirrors `render_reliability.render`: a macro panel spanning a row above the per-domain panels would
    be taller than the 9.25 in text block at `width=\\textwidth`.

    Every panel is square and shares (0,1) limits, so a curve's distance above the chance diagonal is
    comparable by eye. All model curves are SOLID: `plot_color` separates the one same-hue pair by shade,
    and dash is reserved for the chance reference — the same reasoning as the reliability figure, where
    dash marks the perfect-calibration line.
    """
    from matplotlib.lines import Line2D

    domains = [d for d in DOMAINS if any(d in v for v in curves.values())]
    has_macro = any(MACRO_KEY in v for v in curves.values())
    if not domains and not has_macro:
        print("[roc] nothing to plot", file=sys.stderr)
        return

    def draw(ax, d) -> None:
        ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=GREY["ref"], zorder=1)
        for key, per in curves.items():
            c = per.get(d)
            if c is None or not np.isfinite(c["tpr"]).any():
                continue
            col = plot_color(ROSTER[key])
            ax.plot(grid, c["tpr"], "-", color=col, lw=1.4, zorder=3, alpha=0.95)
            if show_ci:
                ax.fill_between(grid, c["lo"], c["hi"], color=col, alpha=0.13, lw=0, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=FONT["tick"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    panels = list(domains) + ([MACRO_KEY] if has_macro else [])
    L_, R_, GAP_ = 0.52, 0.10, 0.26
    pw = (TEXT_WIDTH_IN - L_ - R_ - (len(panels) - 1) * GAP_) / len(panels)
    TOP, XLAB, LEG = 0.30, 0.42, 0.62
    fig = new_figure(TOP + pw + XLAB + LEG)
    axes = panel_row(fig, len(panels), y=LEG + XLAB, h=pw, left=L_, right=R_, gap=GAP_)

    for ax, d in zip(axes, panels):
        draw(ax, d)
        ax.set_title("All domains (macro)" if d == MACRO_KEY else dn.domain(d), fontsize=FONT["title"])
        ax.set_xlabel("false-positive rate", fontsize=FONT["label"])
    axes[0].set_ylabel("true-positive rate", fontsize=FONT["label"])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # Legend carries the MACRO AUROC, the headline number; per-domain values are in the .md/.json.
    handles = [
        Line2D(
            [],
            [],
            color=plot_color(ROSTER[k]),
            ls="-",
            lw=1.6,
            label=f"{ROSTER[k].label}  {curves[k].get(MACRO_KEY, {}).get('auroc', float('nan')):.3f}",
        )
        for k in curves
    ]
    handles.append(Line2D([], [], color=GREY["ref"], ls="--", lw=1.1, label="chance"))
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
    save_figure(fig, out_base, tag="roc", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None, help="override the eval tree root")
    ap.add_argument("--keys", default=None, help="comma list of roster keys (default: full roster order)")
    ap.add_argument(
        "--tag",
        default="primary",
        help="cells to read: 'primary' (default, one forecast per episode), a single 'ctx<k>', or 'ctx*' "
        "to pool the sensitivity sweep. This -- not --testset -- selects the config.",
    )
    ap.add_argument("--out-dir", default="outputs/roc")
    ap.add_argument("--n-boot", type=int, required=True, help="bootstrap draws for the TPR band; 0 = curves only")
    ap.add_argument("--auc-boot", type=int, required=True, help="draws for the macro-AUROC CI in the table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ci",
        action="store_true",
        help="shade the 95%% TPR band. Off by default: with the full roster overlaid the bands occlude each "
        "other; most useful with --keys narrowed to two or three rows.",
    )
    args = ap.parse_args()

    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else list(ORDER)
    if unknown := [k for k in keys if k not in ROSTER]:
        raise SystemExit(f"unknown roster keys {unknown}; known: {list(ORDER)}")

    grid = np.linspace(0.0, 1.0, N_GRID)
    curves, table = {}, []
    for key in keys:
        cells = load_cells(slug, ROSTER[key].eval_key, root=root, tag=args.tag)
        if not cells:
            print(f"[roc] skip {key}: no '{args.tag}' cells", file=sys.stderr)
            continue
        if missing := [d for d in DOMAINS if d not in cells]:
            print(f"[roc] {ROSTER[key].label}: missing {missing} — macro covers the rest", file=sys.stderr)
        curves[key] = curves_for(cells, grid, args.n_boot, args.seed)
        # Macro-AUROC CI through the shared bootstrap, so it matches the master table's scheme.
        flat = [{**r, "domain": d} for d, rows in cells.items() for r in rows]
        pt, lo, hi = macro_ci(flat, auroc_of, n_boot=args.auc_boot, seed=args.seed, stat_id="roc:auroc")
        table.append((ROSTER[key].label, curves[key], pt, lo, hi, {d: len(v) for d, v in cells.items()}))
        print(
            f"[roc] {ROSTER[key].label:24s} macro AUROC={pt:.3f} [{lo:.3f}, {hi:.3f}]  "
            + "  ".join(f"{d}={curves[key][d]['auroc']:.3f}" for d in DOMAINS if d in curves[key])
        )
    if not curves:
        raise SystemExit(f"no '{args.tag}' cells under {eval_paths.result_dir(slug, '', root=root).parent}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "roc_curves.json").write_text(
        json.dumps(
            {
                "testset": args.testset,
                "revision": args.revision,
                "tag": args.tag,
                "fpr_grid": grid.tolist(),
                "rows": {
                    k: {
                        "label": ROSTER[k].label,
                        "panels": {
                            d: {"tpr": np.where(np.isfinite(c["tpr"]), c["tpr"], None).tolist(), "auroc": c["auroc"]}
                            for d, c in v.items()
                        },
                    }
                    for k, v in curves.items()
                },
            },
            indent=2,
        )
    )
    lines = [
        "| Model | macro AUROC [95% CI] | " + " | ".join(dn.domain(d) for d in DOMAINS) + " | n/domain |",
        "|---|---|" + "|".join(["---"] * (len(DOMAINS) + 1)) + "|",
    ]
    for lbl, per, pt, lo, hi, ns in table:
        per_d = " | ".join(f"{per[d]['auroc']:.3f}" if d in per else "—" for d in DOMAINS)
        lines.append(
            f"| {lbl} | {pt:.3f} [{lo:.3f}, {hi:.3f}] | {per_d} | "
            + ", ".join(f"{d}={n}" for d, n in sorted(ns.items()))
            + " |"
        )
    lines += [
        "",
        "AUROC is the tie-corrected Mann-Whitney statistic from `scripts/lib/calibration_metrics.py` — the "
        "same one the master table reports, not the area under the plotted (interpolated) curve.",
        "",
        "`macro AUROC` = mean of the per-domain AUROCs, with a 95% CI from a bootstrap that resamples "
        "episodes within domain and re-macro-averages each draw. The macro *curve* in the figure is a "
        "separate object — per-domain TPR averaged on a shared FPR grid — so its enclosed area need not "
        "equal this number exactly; this number is the one to quote.",
        "",
        "0.5 is chance. A row well above the diagonal while its reliability curve sits far off the "
        "identity is ranking correctly but scoring miscalibrated probabilities — the case Brier alone "
        "cannot separate.",
    ]
    (out_dir / "roc_auc.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    render(curves, grid, out_dir / "roc", show_ci=args.ci, tag=args.tag)
    print(f"[roc] data -> {out_dir / 'roc_curves.json'}, table -> {out_dir / 'roc_auc.md'}")


if __name__ == "__main__":
    main()
