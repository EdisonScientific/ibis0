"""Render the leave-one-out cross-domain transfer matrix (stage 5 of the eval pipeline).

Consumes the per-domain result JSONs written by `eval_classification_testset.py` and emits the 4x4
train-domain x eval-domain matrix:

- **rows** = models: `leave-out-tahoe`, `leave-out-depmap`, `leave-out-gwas`, `all`
- **columns** = test domains: `tahoe`, `depmap`, `gwas`, plus `all` = the **macro-average** over the three

Both axes use the same order (tahoe, depmap, gwas, then `all`), so row `i` lines up with column `i` and
every transfer cell falls on the **diagonal**.

Each leave-one-out row has exactly one **transfer cell** — the column for the domain it never trained
on. Those cells are the scientific payload; the rest are in-domain references, and the `all` row is the
upper reference that trained on everything.

Three things this script is careful about:

1. **Macro, not micro.** The `all` column averages the three per-domain values with equal weight, and
   its CI comes from a single stratified bootstrap in which the macro-average *is* the statistic (via
   `scripts/lib/macro_bootstrap.py`) — not from combining three per-domain CIs, which has no valid
   interpretation. Per-cell CIs come from the same machinery restricted to one domain.
2. **Transfer gaps are paired.** The cost of never seeing a domain is `LOO_model - all_model` on that
   domain. Because every model scored byte-identical episodes (guaranteed by the published test set),
   the difference is bootstrapped with a *shared* per-iteration resample, which cancels
   shared-episode variance and gives a much tighter CI on the gap than differencing two independent CIs.
3. **Lower-is-better vs higher-is-better.** Brier/ECE are losses, AUROC/accuracy are scores; the sign of
   a "good" transfer gap flips accordingly, so the metric's direction is carried explicitly rather than
   assumed.

Outputs (all under `--out-dir`):
  transfer_matrix_<metric>.md    human-readable table + the paired transfer gaps
  transfer_matrix_<metric>.csv   tidy long form (row, column, value, lo, hi, n, is_transfer)
  transfer_matrix_<metric>.pdf   annotated heatmap (the paper artifact)
  transfer_matrix_<metric>.png   same figure for quick viewing
  transfer_combined.pdf          compact Brier matrix + transfer breakdown (`--combined-figure`)
  transfer_combined.png          raster companion for quick viewing
  transfer_full.tex              complete appendix table: Brier matrix + held-out decomposition
  cell_metrics.json              the FULL metric suite per cell (point estimates), so nothing is lost

Note on the palette: `scripts/lib/roster_palette.py` assigns a colour per *model*, which suits the
model-series figures (grouped metrics, reliability, dose). A matrix carries model identity in its row
labels, so this figure uses a sequential colormap for the metric value instead.

Example — after the four matrix jobs land (render_all.sh runs it this way):
    uv run python scripts/analysis/multidomain/render_transfer_matrix.py --combined-figure \\
        --n-boot 2000 --out-dir outputs/transfer_matrix
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

_SCRIPTS = next(p for p in Path(__file__).resolve().parents if p.name == "scripts")
import sys  # noqa: E402

sys.path.insert(0, str(_SCRIPTS / "lib"))
sys.path.insert(0, str(_SCRIPTS.parent / "envs" / "ibis_env_utils"))
import display_names as names  # noqa: E402
import eval_paths as ep  # noqa: E402
from calibration_metrics import compute_calibration  # noqa: E402
from figure_io import FONT, GREY, TEXT_WIDTH_IN, new_figure, save_figure  # noqa: E402
from macro_bootstrap import macro_ci, paired_macro_ci  # noqa: E402
from roster_palette import ROSTER  # noqa: E402

DOMAINS = ["tahoe", "depmap", "gwas"]


# (row label, run-id, domain held out of training). `all` holds nothing out.
#
# Row order mirrors DOMAINS, with `all` last on both axes. That alignment is deliberate: row i then
# corresponds to column i, so every transfer cell lands on the **diagonal** and the off-diagonal is
# uniformly in-domain — far easier to read than a scattered arrangement, in the table and the heatmap.
class Row(NamedTuple):
    """One matrix row: a trained model, identified the way `eval_paths` keys it."""

    label: str
    run_name: str
    step: int
    held_out: str | None  # the column that is this row's zero-shot transfer cell
    # Set for rows that are not checkpoints. The untrained base has no run/step, but it is the reference
    # every retention figure is measured against, so it has to load through the same path as the rest.
    eval_key: str | None = None


ROWS: list[Row] = [
    Row("leave-out-tahoe", "gwas-depmap__qwen3-30b-a3b__cls__r2.1.0", 1700, "tahoe"),
    Row("leave-out-depmap", "gwas-tahoe__qwen3-30b-a3b__cls__r2.1.0", 1300, "depmap"),
    Row("leave-out-gwas", "depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0", 1800, "gwas"),
    Row("all", "gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0", 2400, None),
]

# The two REFERENCE rows retention is computed between. Neither is drawn as a matrix row: they are the
# endpoints of the interval each cell is placed within.
#
#   retention = (base - model) / (base - all)      on that domain
#
# 100% = the model does as well as one that trained on the domain; 0% = no better than never training at
# all; negative = training on the OTHER domains actively hurt. The ratio is direction-agnostic — for a
# higher-is-better metric both differences flip sign and cancel — so it works unchanged for resolution
# and AUROC as well as Brier.
#
# `base (direct)` is the untrained same model, which makes the denominator "what did this training buy on
# this domain" rather than "how does this compare to some other system".
BASE_LABEL, ALL_LABEL = "base", "all"
BASE_ROW = Row(BASE_LABEL, "", 0, None, eval_key=ROSTER["base_direct"].eval_key)


def _col_label(c: str) -> str:
    """Column heading. The `all` column is a macro-average, not a domain, so say so."""
    return "All (macro)" if c == "all" else names.domain(c)


def _statistic(metric: str):
    """A `list[record] -> float` statistic, for both point estimates and bootstrap draws.

    Kept cheap: brier/accuracy are closed-form, and the `compute_calibration` path disables sMECE
    (a kernel estimate) since it would otherwise dominate a 2000-iteration bootstrap.
    """

    def stat(rows: list) -> float:
        pl = [(r["parsed_prob"], r["label"]) for r in rows if r.get("parsed_prob") is not None]
        if not pl:
            return float("nan")
        probs = np.array([p for p, _ in pl], dtype=float)
        labels = np.array([y for _, y in pl], dtype=float)
        if metric == "brier":
            return float(np.mean((probs - labels) ** 2))
        if metric == "accuracy":
            return float(np.mean((probs >= 0.5) == (labels >= 0.5)))
        rep = compute_calibration(probs, labels, with_smece=(metric == "smece"))
        v = getattr(rep, metric)
        return float("nan") if v is None else float(v)

    return stat


def load_cells(testset_slug: str, rows: list[Row], tag: str, root: Path | None = None):
    """{row_label: {domain: [record, ...]}} from the canonical eval tree.

    Paths come from `eval_paths`, the same module the eval writers use, so the reader cannot drift from
    the layout. Each record is stamped with its `domain` so the bootstrap can stratify on it, and with
    its `episode_id` so the paired bootstrap can align models on shared episodes.
    """
    cells: dict[str, dict[str, list]] = {}
    for row in rows:
        label = row.label
        cells[label] = {}
        key = row.eval_key or ep.checkpoint_key(row.run_name, row.step)
        for d in DOMAINS:
            p = ep.cell_path(testset_slug, key, d, tag, root=root)
            if not p.exists():
                print(f"  MISSING {label}/{d}: {p}")
                continue
            payload = json.loads(p.read_text())
            recs = payload["results"]
            for r in recs:
                r.setdefault("domain", d)  # older writers omitted it; stratification needs it
            if payload.get("domain") not in (None, d):
                raise ValueError(f"{p}: header domain={payload['domain']!r} but filename says {d!r}")
            cells[label][d] = recs
    return cells


def _fmt(v: float, lo: float | None = None, hi: float | None = None, nd: int = 4) -> str:
    if not np.isfinite(v):
        return "—"
    if lo is None or not np.isfinite(lo):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def build(cells, rows, metric: str, n_boot: int, seed: int):
    """Per-cell values with CIs, plus the macro `all` column per row."""
    stat = _statistic(metric)
    out: dict[str, dict[str, dict]] = {}
    for label, held_out in ((r.label, r.held_out) for r in rows):
        out[label] = {}
        per_domain = cells.get(label, {})
        for d in DOMAINS:
            recs = per_domain.get(d)
            if not recs:
                out[label][d] = {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
                continue
            # macro_ci over a single domain reduces to an ordinary within-domain bootstrap.
            v, lo, hi = macro_ci(recs, stat, n_boot=n_boot, seed=seed, stat_id=f"transfer:{metric}:per_domain")
            out[label][d] = {
                "value": v,
                "lo": lo,
                "hi": hi,
                "n": len(recs),
                "is_transfer": d == held_out,
            }
        pooled = [r for d in DOMAINS for r in per_domain.get(d, [])]
        if pooled and all(per_domain.get(d) for d in DOMAINS):
            v, lo, hi = macro_ci(pooled, stat, n_boot=n_boot, seed=seed, stat_id=f"transfer:{metric}:macro")
            out[label]["all"] = {"value": v, "lo": lo, "hi": hi, "n": len(pooled), "is_transfer": False}
        else:
            out[label]["all"] = {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return out


def add_retention(matrix: dict, rows: list[Row]) -> dict:
    """Annotate each displayed cell with `retention`, its position between base and in-domain training.

    See BASE_ROW for the definition. Guards a near-zero denominator: when training bought essentially
    nothing on a domain, "what fraction of that did the transfer model capture" has no meaningful answer,
    and a ratio of two tiny differences would be numerical noise presented as a percentage.
    """
    base, allm = matrix.get(BASE_LABEL, {}), matrix.get(ALL_LABEL, {})
    for row in rows:
        for d in DOMAINS:
            cell = matrix.get(row.label, {}).get(d)
            if cell is None:
                continue
            b = base.get(d, {}).get("value", float("nan"))
            a = allm.get(d, {}).get("value", float("nan"))
            span = b - a
            cell["retention"] = (b - cell["value"]) / span if np.isfinite(span) and abs(span) > 1e-6 else float("nan")
    return matrix


# Which side of the Murphy decomposition a metric reports. This is the figure's whole claim, so it is
# declared rather than inferred: calibration terms transfer, discrimination terms do not, and Brier mixes
# them, which is why the composite reads as weak transfer.
METRIC_FAMILY = {
    "brier": "composite",
    "reliability": "calibration",
    "smece": "calibration",
    "ece": "calibration",
    "resolution": "discrimination",
    "auroc": "discrimination",
    "auprc": "discrimination",
}
# Deliberately NOT roster colours: these encode a property of the metric, and reusing the model palette
# would invite reading a green dot as a model.
FAMILY_COLOR = {"calibration": "#1b7837", "discrimination": "#762a83", "composite": "#404040"}


def retention_ci(cells, rows: list[Row], metric: str, n_boot: int, seed: int) -> list[dict]:
    """[{held_out, metric, point, lo, hi}] — retention on each row's held-out domain, with a PAIRED CI.

    Paired matters here more than usual. Retention is a ratio of two differences between three models
    scored on the SAME episodes, so base/LOO/all-domain errors are strongly correlated; three independent
    bootstraps would inflate the interval by resampling that shared variance three times over. Drawing the
    episodes once per iteration and applying that draw to all three cancels it.

    A draw whose denominator collapses yields NaN rather than an exploding ratio, and `paired_macro_ci`
    takes nan-percentiles, so a degenerate draw drops out instead of setting the interval.
    """
    stat = _statistic(metric)
    out = []
    for row in rows:
        d = row.held_out
        by_model = {
            "base": [r for r in cells.get(BASE_LABEL, {}).get(d, [])],
            "loo": [r for r in cells.get(row.label, {}).get(d, [])],
            "all": [r for r in cells.get(ALL_LABEL, {}).get(d, [])],
        }
        if not all(by_model.values()):
            continue

        def combine(m):
            span = m["base"] - m["all"]
            return (m["base"] - m["loo"]) / span if abs(span) > 1e-6 else float("nan")

        pt, lo, hi = paired_macro_ci(
            by_model,
            stat,
            combine,
            n_boot=n_boot,
            seed=seed,
            stat_id=f"transfer:retention:{metric}:{d}",
        )
        # The three levels retention is computed FROM. Retention says where the model sits between them
        # but not what the interval spans, and 62% of a large gap is a different claim from 62% of a
        # negligible one — so the absolutes travel with the ratio rather than being recoverable only
        # from another file.
        out.append(
            {
                "held_out": d,
                "metric": metric,
                "point": pt,
                "lo": lo,
                "hi": hi,
                "abs_base": stat(by_model["base"]),
                "abs_model": stat(by_model["loo"]),
                "abs_all": stat(by_model["all"]),
            }
        )
    return out


def transfer_losses(cells, rows: list[Row], metric: str, n_boot: int, seed: int) -> list[dict]:
    """Direction-normalized paired loss from holding out each domain; positive is always worse."""
    stat = _statistic(metric)
    lower_better = names.lower_is_better(metric)
    out = []
    for row in rows:
        d = row.held_out
        loo = cells.get(row.label, {}).get(d)
        full = cells.get(ALL_LABEL, {}).get(d)
        if not loo or not full:
            continue

        def combine(values):
            return values["loo"] - values["full"] if lower_better else values["full"] - values["loo"]

        point, lo, hi = paired_macro_ci(
            {"loo": loo, "full": full},
            stat,
            combine,
            n_boot=n_boot,
            seed=seed,
            stat_id=f"transfer:loss_vs_full:{metric}:{d}",
        )
        out.append({"held_out": d, "metric": metric, "point": point, "lo": lo, "hi": hi})
    return out


def _latex_estimate(entry: dict, *, ndigits: int = 4, bold: bool = False) -> str:
    value = entry.get("value", float("nan"))
    lo = entry.get("lo", float("nan"))
    hi = entry.get("hi", float("nan"))
    if not np.isfinite(value):
        return r"\textemdash"
    point = f"{value:.{ndigits}f}"
    if bold:
        point = rf"\textbf{{{point}}}"
    if not np.isfinite(lo) or not np.isfinite(hi):
        return point
    return rf"\tmstack{{{point}}}{{[{lo:.{ndigits}f}, {hi:.{ndigits}f}]}}"


def _latex_interval(point: float, lo: float, hi: float, *, ndigits: int = 4, percent: bool = False) -> str:
    if not all(np.isfinite(v) for v in (point, lo, hi)):
        return r"\textemdash"
    scale = 100.0 if percent else 1.0
    suffix = r"\%" if percent else ""
    return rf"\tmstack{{{point * scale:.{ndigits}f}{suffix}}}{{[{lo * scale:.{ndigits}f}, {hi * scale:.{ndigits}f}]{suffix}}}"


def render_full_transfer_latex(
    per_metric: dict[str, dict],
    rows: list[Row],
    retention: dict[str, list[dict]],
    losses: dict[str, list[dict]],
) -> str:
    """Two-block appendix table: complete Brier matrix and exact held-out transfer decomposition."""
    required_metrics = ("brier", "reliability", "resolution")
    missing = [metric for metric in required_metrics if metric not in per_metric]
    if missing:
        raise ValueError(f"full transfer table requires {required_metrics}; missing {missing}")

    loo_rows = [row for row in rows if row.held_out]
    all_row = next((row for row in rows if row.label == ALL_LABEL), None)
    if len(loo_rows) != len(DOMAINS) or all_row is None:
        raise ValueError("full transfer table requires all three leave-one-out rows and the all-domain row")

    training_mix = {
        "tahoe": "DepMap + GWAS (Tahoe held out)",
        "depmap": "Tahoe + GWAS (DepMap held out)",
        "gwas": "Tahoe + DepMap (GWAS held out)",
    }
    lines = [
        r"\providecommand{\tmstack}[2]{\begin{tabular}[t]{@{}c@{}}#1\\[-2pt]{\scriptsize\color{black!45}#2}\end{tabular}}%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\multicolumn{5}{@{}l}{\textbf{A. Absolute Brier performance}} \\[2pt]",
        r"\toprule",
        r"Training mix & Tahoe & DepMap & GWAS & Macro \\",
        r"\midrule",
    ]
    brier = per_metric["brier"]
    for row in loo_rows:
        cells = [_latex_estimate(brier[row.label][domain], bold=(domain == row.held_out)) for domain in DOMAINS]
        cells.append(_latex_estimate(brier[row.label]["all"]))
        lines.append(f"{training_mix[row.held_out]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    all_cells = [_latex_estimate(brier[ALL_LABEL][domain]) for domain in DOMAINS + ["all"]]
    lines.extend(
        [
            "All domains & " + " & ".join(all_cells) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.8em}",
            r"\begin{tabular}{@{}llccccc@{}}",
            r"\multicolumn{7}{@{}l}{\textbf{B. Held-out-domain transfer decomposition}} \\[2pt]",
            r"\toprule",
            r"Held-out & Metric & Qwen3 & Leave-one-out & Full model & Loss vs. full & Gain retained \\",
            r"\midrule",
        ]
    )

    retention_by_metric = {
        metric: {entry["held_out"]: entry for entry in retention[metric]} for metric in required_metrics
    }
    losses_by_metric = {metric: {entry["held_out"]: entry for entry in losses[metric]} for metric in required_metrics}
    row_by_domain = {row.held_out: row for row in loo_rows}
    for domain_index, domain in enumerate(DOMAINS):
        row = row_by_domain[domain]
        for metric_index, metric in enumerate(required_metrics):
            matrix = per_metric[metric]
            loss = losses_by_metric[metric][domain]
            retained = retention_by_metric[metric][domain]
            domain_text = names.domain(domain) if metric_index == 0 else ""
            metric_text = names.metric(metric)
            cells = [
                _latex_estimate(matrix[BASE_LABEL][domain]),
                _latex_estimate(matrix[row.label][domain]),
                _latex_estimate(matrix[ALL_LABEL][domain]),
                _latex_interval(loss["point"], loss["lo"], loss["hi"]),
                _latex_interval(retained["point"], retained["lo"], retained["hi"], ndigits=1, percent=True),
            ]
            lines.append(f"{domain_text} & {metric_text} & " + " & ".join(cells) + r" \\")
        if domain_index < len(DOMAINS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_retention_dots(per_metric: dict[str, list[dict]], path: Path) -> None:
    """Retention on each held-out domain, one dot-and-whisker per (domain, metric).

    A 3x3-per-metric grid would spend 27 cells on 9 numbers: only the diagonal of that grid is a
    transfer cell, and the off-diagonal is ~100% everywhere by construction. That off-diagonal is a
    validity check — a model that DID train on a domain should score ~100%, and it does — so it belongs in
    the sidecar and the caption rather than in 18 cells of figure.

    Two reference rules carry the interpretation: 0% is "no better than never training on the domain",
    100% is "as good as training on it". A metric landing left of 0 means training on the OTHER domains
    actively hurt, which happens for GWAS discrimination and which a colour scale would have flattened.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    metrics = list(per_metric)
    domains = [d for d in DOMAINS if any(e["held_out"] == d for m in metrics for e in per_metric[m])]
    # METRIC-major, domain-minor: one block per metric, the three held-out domains within it. The metric
    # is what the figure contrasts, so it reads as the grouping; and because a whole block shares one
    # metric it shares one family colour, which makes the calibration/discrimination split a property of
    # the blocks rather than something to decode dot by dot.
    #
    # It also resolves a label collision. Domain-major put the long metric names on the y ticks, and the
    # block header ran into them; metric-major puts the SHORT domain names there and the long name in the
    # header, where it has the full width.
    rows_spec = [(m, d) for m in metrics for d in domains]

    L_, R_ = 0.72, 0.62
    ROW_H, GROUP_PAD = 0.34, 0.52
    BAR_H = 0.66  # bar thickness in row units
    LABEL_X_IN = 0.10  # left edge shared by the row labels and the block headers
    body = len(rows_spec) * ROW_H + (len(metrics) - 1) * GROUP_PAD
    TOP, BOT, LEG = 0.34, 0.40, 0.40
    figh = TOP + body + BOT + LEG
    fig = new_figure(figh)
    ax = fig.add_axes((L_ / TEXT_WIDTH_IN, (BOT + LEG) / figh, (TEXT_WIDTH_IN - L_ - R_) / TEXT_WIDTH_IN, body / figh))

    ypos, ylabels = [], []
    y = 0.0
    for gi, m in enumerate(metrics):
        for d in domains:
            ypos.append(y)
            ylabels.append(names.domain(d))
            y += 1.0
        if gi < len(metrics) - 1:
            y += GROUP_PAD / ROW_H

    for rule, style in ((0.0, "-"), (1.0, "--")):
        ax.axvline(rule, color=GREY["ref"], ls=style, lw=1.0, zorder=1)

    for (m, d), yy in zip(rows_spec, ypos):
        e = next((x for x in per_metric[m] if x["held_out"] == d), None)
        if e is None:
            continue
        col = FAMILY_COLOR[METRIC_FAMILY.get(m, "composite")]
        # The interval as a filled BAR rather than a hairline, with the point estimate as a rule through
        # it. Nine hairlines left the panel mostly empty, and interval width is the thing being compared,
        # so it should carry visual weight proportional to that.
        #
        # NOT a box plot, despite the resemblance: a box plot's box is the interquartile range of a
        # distribution, whereas this is a 95% confidence interval on a single statistic. Same shape,
        # different meaning — hence the legend says so rather than leaving the idiom to imply quartiles.
        if np.isfinite(e["lo"]) and np.isfinite(e["hi"]):
            ax.barh(
                yy,
                e["hi"] - e["lo"],
                left=e["lo"],
                height=BAR_H,
                color=col,
                alpha=0.30,
                edgecolor=col,
                linewidth=0.8,
                zorder=3,
            )
        ax.plot([e["point"]] * 2, [yy - BAR_H / 2, yy + BAR_H / 2], color=col, lw=1.8, zorder=4)
        # Value at the RIGHT END of the bar, vertically centred. Above the bar it collided with the bar
        # in the row above once the bars got thick — and inside the bar it does not fit the narrow ones
        # (Tahoe reliability spans 6 points). To the right it never collides regardless of bar width.
        # Value and ratio trail the bar, so they move with the data rather than forming a side column --
        # a column reads as its own table with its own row order. The base -> all interval they are a
        # fraction of lives under the ROW LABEL instead: it is the same pair for every estimate in that
        # row, so repeating it beside each bar would suggest it varied with the estimate.
        ax.text(
            max(e["hi"], e["point"]) + 0.015,
            yy,
            f"{e['abs_model']:.3f} ({e['point']:.0%})",
            ha="left",
            va="center",
            fontsize=FONT["annot"],
            color=col,
            # Scrim, because a bar ending near 90% pushes its label across the 100% rule and the dashed
            # line would otherwise strike through the digits. Same treatment as the headline figure's
            # in-panel reference labels.
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.78),
        )

    # Row labels are drawn by hand rather than as tick labels: a y tick label is right-aligned against
    # the axis and cannot be left-aligned without its text growing into the plot area. Drawn at a fixed x
    # they share a left edge with the block headers, so both columns of text line up.
    ax.set_yticks(ypos, [""] * len(ypos))
    ax.invert_yaxis()
    label_x = (LABEL_X_IN - L_) / (TEXT_WIDTH_IN - L_ - R_)  # inches from the canvas -> axes coords
    for (m, d), yy, lbl in zip(rows_spec, ypos, ylabels):
        ax.text(
            label_x,
            yy - 0.20,
            lbl,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FONT["annot"],
            color=GREY["ink"],
        )
        # Second line: the interval the retention is a fraction of. It belongs to the ROW — the same two
        # endpoints for every estimate in it — so it sits with the row label rather than trailing the bar,
        # where it would be re-read as though it varied with the estimate.
        e = next((x for x in per_metric[m] if x["held_out"] == d), None)
        if e is not None:
            ax.text(
                label_x,
                yy + 0.22,
                f"{e['abs_base']:.3f}→{e['abs_all']:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=FONT["annot"],
                color=GREY["note"],
                style="italic",
            )
    ax.set_xlabel("retention of the achievable gain, on the domain held out of training", fontsize=FONT["label"])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(labelsize=FONT["tick"])
    # No y tick marks: the labels are drawn by hand well to the left, so a tick would float in the
    # gap between them and the axes, pointing at nothing.
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.18)

    # Metric header above each block, sharing the row labels' left edge. Two separate texts rather than
    # one two-line string: the name is bold and the family is italic, which a single call cannot express.
    for gi, m in enumerate(metrics):
        top = ypos[gi * len(domains)]
        ax.text(
            label_x,
            top - 1.30,
            f"{names.metric(m)} ({names.direction_arrow(m)})",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FONT["annot"],
            color=FAMILY_COLOR[METRIC_FAMILY.get(m, "composite")],
            fontweight="bold",
        )
        ax.text(
            label_x,
            top - 0.88,
            METRIC_FAMILY.get(m, "composite"),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FONT["annot"],
            color=FAMILY_COLOR[METRIC_FAMILY.get(m, "composite")],
            style="italic",
        )

    # Family is already stated in each block header and carried by its colour, so the legend carries only
    # the two reference rules — repeating the family names here would be a third encoding of one fact.
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=GREY["note"], alpha=0.30, edgecolor=GREY["note"], label="95% CI (paired bootstrap)"),
        Line2D([], [], color=GREY["note"], lw=1.8, label="point estimate"),
        Line2D([], [], color=GREY["ref"], ls="-", lw=1.0, label="0% = never trained on this domain"),
        Line2D([], [], color=GREY["ref"], ls="--", lw=1.0, label="100% = trained on this domain"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=FONT["legend"],
        handlelength=1.6,
        columnspacing=1.8,
        bbox_to_anchor=(0.5, 0.010),
    )
    save_figure(fig, path, tag="transfer:retention-dots", tight=False)


def render_combined_transfer(
    brier_matrix: dict[str, dict],
    rows: list[Row],
    per_metric: dict[str, list[dict]],
    path: Path,
) -> None:
    """Compact experiment matrix and the existing transfer-retention summary, side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required_metrics = ("brier", "reliability", "resolution")
    missing = [m for m in required_metrics if m not in per_metric]
    if missing:
        raise ValueError(f"combined transfer figure requires {required_metrics}; missing {missing}")

    loo_rows = [r for r in rows if r.held_out]
    if len(loo_rows) != len(DOMAINS):
        raise ValueError("combined transfer figure requires all three leave-one-domain-out rows")

    figh = 2.70
    fig = new_figure(figh)
    second_line_font = 4.7

    # Panel A: the row labels state the data the classifier did see, while the parenthetical makes the
    # held-out domain explicit. Keeping only the 3x3 experiment grid avoids treating the all-domain
    # reference as a fourth training condition.
    panel_y = 0.48
    panel_h = 1.75
    ax_a = fig.add_axes((1.02 / TEXT_WIDTH_IN, panel_y / figh, panel_h / TEXT_WIDTH_IN, panel_h / figh))
    vals = np.array([[brier_matrix[r.label][domain]["value"] for domain in DOMAINS] for r in loo_rows], dtype=float)
    finite = vals[np.isfinite(vals)]
    im = ax_a.imshow(
        vals,
        cmap="viridis_r",
        vmin=(finite.min() if finite.size else 0),
        vmax=(finite.max() if finite.size else 1),
        aspect="auto",
        interpolation="nearest",
    )
    seen_domains = {
        "tahoe": "DepMap + GWAS",
        "depmap": "Tahoe + GWAS",
        "gwas": "Tahoe + DepMap",
    }
    ax_a.set_xticks(range(len(DOMAINS)), [names.domain(d) for d in DOMAINS], fontsize=FONT["tick"])
    for label in ax_a.get_xticklabels():
        label.set_fontweight("normal")
    ax_a.set_yticks(range(len(loo_rows)), [""] * len(loo_rows))
    for i, row in enumerate(loo_rows):
        ax_a.text(
            -0.08,
            i - 0.07,
            seen_domains[row.held_out],
            transform=ax_a.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=FONT["annot"],
            fontweight="normal",
            color=GREY["ink"],
        )
        ax_a.text(
            -0.08,
            i + 0.10,
            f"({names.domain(row.held_out)} held-out)",
            transform=ax_a.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=second_line_font,
            color=GREY["note"],
        )
    ax_a.set_xlabel("Test", fontsize=FONT["label"], fontweight="normal")
    ax_a.set_ylabel("Train", fontsize=FONT["label"], fontweight="normal")
    ax_a.yaxis.set_label_coords(-0.52, 0.5)
    ax_a.tick_params(length=0)
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    lo, hi = im.get_clim()
    for i, row in enumerate(loo_rows):
        for j, domain in enumerate(DOMAINS):
            entry = brier_matrix[row.label][domain]
            rel = (entry["value"] - lo) / max(hi - lo, 1e-12)
            colour = "white" if rel > 0.55 else "black"
            if domain == row.held_out:
                retention = entry.get("retention", float("nan"))
                ax_a.text(
                    j,
                    i - 0.14,
                    f"{entry['value']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=FONT["annot"],
                    fontweight="bold",
                    color=colour,
                )
                detail = "OOD" if not np.isfinite(retention) else f"OOD ({retention:.0%})"
                ax_a.text(
                    j,
                    i + 0.20,
                    detail,
                    ha="center",
                    va="center",
                    fontsize=second_line_font,
                    color="white",
                    style="italic",
                )
                ax_a.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        ec=GREY["flag"],
                        lw=1.8,
                        zorder=5,
                        clip_on=False,
                    )
                )
            else:
                ax_a.text(
                    j,
                    i,
                    f"{entry['value']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=FONT["annot"],
                    color=colour,
                )

    cax = fig.add_axes((2.82 / TEXT_WIDTH_IN, panel_y / figh, 0.075 / TEXT_WIDTH_IN, panel_h / figh))
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    colour_ticks = np.linspace(lo, hi, 5)
    cb.set_ticks(colour_ticks, labels=[f"{value:.3f}" for value in colour_ticks])
    cb.ax.tick_params(labelsize=FONT["annot"], length=2, pad=1)
    cb.outline.set_visible(False)
    # Panel B retains the standalone plot's intervals, point markers, and 0%/100% references, while
    # labeling only the normalized retention values needed for the cross-component comparison.
    metrics = list(required_metrics)
    domains = list(DOMAINS)
    rows_spec = [(metric, domain) for metric in metrics for domain in domains]
    ax_b = fig.add_axes((3.75 / TEXT_WIDTH_IN, panel_y / figh, 2.79 / TEXT_WIDTH_IN, panel_h / figh))
    group_pad = 0.52 / 0.34
    ypos: list[float] = []
    y = 0.0
    for gi, _metric in enumerate(metrics):
        for _domain in domains:
            ypos.append(y)
            y += 1.0
        if gi < len(metrics) - 1:
            y += group_pad

    for rule, style in ((0.0, "-"), (1.0, "--")):
        ax_b.axvline(rule, color=GREY["ref"], ls=style, lw=1.0, zorder=1)

    bar_h = 0.66
    for (metric, domain), yy in zip(rows_spec, ypos):
        entry = next(x for x in per_metric[metric] if x["held_out"] == domain)
        colour = FAMILY_COLOR[METRIC_FAMILY.get(metric, "composite")]
        if np.isfinite(entry["lo"]) and np.isfinite(entry["hi"]):
            ax_b.barh(
                yy,
                entry["hi"] - entry["lo"],
                left=entry["lo"],
                height=bar_h,
                color=colour,
                alpha=0.30,
                edgecolor=colour,
                linewidth=0.8,
                zorder=3,
            )
        ax_b.plot([entry["point"]] * 2, [yy - bar_h / 2, yy + bar_h / 2], color=colour, lw=1.8, zorder=4)
        ax_b.text(
            max(entry["hi"], entry["point"]) + 0.015,
            yy,
            f"{entry['point']:.0%}",
            ha="left",
            va="center",
            fontsize=FONT["annot"],
            color=colour,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.78),
        )

    ax_b.set_yticks(ypos, [""] * len(ypos))
    ax_b.invert_yaxis()
    label_x = -0.14
    for (_metric, domain), yy in zip(rows_spec, ypos):
        ax_b.text(
            label_x,
            yy,
            names.domain(domain),
            transform=ax_b.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FONT["annot"],
            color=GREY["ink"],
        )

    for gi, metric in enumerate(metrics):
        top = ypos[gi * len(domains)]
        colour = FAMILY_COLOR[METRIC_FAMILY.get(metric, "composite")]
        ax_b.text(
            label_x,
            top - 1.30,
            names.metric(metric),
            transform=ax_b.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=FONT["annot"],
            color=colour,
            fontweight="bold",
        )

    ax_b.set_xlim(-0.22, 1.05)
    ax_b.set_xlabel("% gain of full model retained", fontsize=FONT["label"])
    ax_b.xaxis.set_label_coords(0.5, -0.11)
    ax_b.xaxis.label.set_horizontalalignment("center")
    ax_b.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax_b.tick_params(labelsize=FONT["tick"])
    ax_b.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax_b.spines[spine].set_visible(False)
    ax_b.grid(axis="x", alpha=0.18)
    ax_b.text(
        0.0,
        1.01,
        "Qwen3 (0%)",
        transform=ax_b.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=FONT["annot"],
        color=GREY["note"],
    )
    ax_b.text(
        1.0,
        1.01,
        "full model (100%)",
        transform=ax_b.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=FONT["annot"],
        color=GREY["note"],
    )

    fig.text(0.80 / TEXT_WIDTH_IN, 0.970, "A", ha="left", va="top", fontsize=FONT["title"], fontweight="bold")
    fig.text(
        1.96 / TEXT_WIDTH_IN,
        0.970,
        "Cross-domain Brier score (↓)",
        ha="center",
        va="top",
        fontsize=FONT["title"],
    )
    fig.text(3.48 / TEXT_WIDTH_IN, 0.970, "B", ha="left", va="top", fontsize=FONT["title"], fontweight="bold")
    fig.text(
        5.15 / TEXT_WIDTH_IN,
        0.970,
        "Transfer breakdown by component metric",
        ha="center",
        va="top",
        fontsize=FONT["title"],
    )
    save_figure(fig, path, tag="transfer:combined", tight=False)


def render_retention_panels(per_metric: dict[str, dict], rows: list[Row], path: Path) -> None:
    """One 3x3 heatmap per metric, side by side. Cell text = absolute value over retention %.

    Rows are the leave-one-out models, columns the test domains, so **the diagonal is the transfer cell**
    in every panel and the off-diagonal is uniformly in-domain. The all-domain model and the untrained
    base do not appear as rows: they are the endpoints retention is measured between, and showing them as
    rows would invite reading the matrix as a four-way comparison rather than as a placement.

    **Colour is retention, not the metric.** Three metrics on three different scales cannot share a
    colourbar, and giving each panel its own would make identical shades mean different things across the
    figure — the exact confusion this figure exists to remove. Retention is already normalised, so one
    scale serves all three and a cell's colour is comparable everywhere. The absolute value is still
    printed, because a normalised number alone hides whether the underlying quantity moved at all.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    metrics = list(per_metric)
    labels = [r.label for r in rows]
    L_, R_, GAP_, CBAR = 1.02, 0.66, 0.20, 0.16
    pw = (TEXT_WIDTH_IN - L_ - R_ - (len(metrics) - 1) * GAP_ - CBAR) / len(metrics)
    cell_h = 0.56
    grid_h = cell_h * len(rows)
    TOP, BOT, CAP = 0.54, 0.34, 0.20
    figh = TOP + grid_h + BOT + CAP
    fig = new_figure(figh)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under(GREY["flag"])  # negative retention: training on the others HURT this domain
    norm = Normalize(vmin=0.0, vmax=1.0)

    for i, metric in enumerate(metrics):
        m = per_metric[metric]
        x0 = L_ + i * (pw + GAP_)
        ax = fig.add_axes((x0 / TEXT_WIDTH_IN, (BOT + CAP) / figh, pw / TEXT_WIDTH_IN, grid_h / figh))
        ret = np.array([[m[lbl][d].get("retention", np.nan) for d in DOMAINS] for lbl in labels], dtype=float)
        ax.imshow(ret, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        for r, lbl in enumerate(labels):
            for c, d in enumerate(DOMAINS):
                e = m[lbl][d]
                rv = e.get("retention", float("nan"))
                # Text contrast follows the CELL it sits on. A NaN cell is left unpainted by
                # imshow, so it must fall back to dark-on-white rather than inheriting the
                # under-colour treatment -- white text on an unpainted cell is invisible.
                if not np.isfinite(rv):
                    col = GREY["ink"]
                else:
                    shade = cmap(norm(rv)) if rv >= 0 else cmap.get_under()
                    lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
                    col = "white" if lum < 0.6 else "black"
                weight = "bold" if e.get("is_transfer") else "normal"
                ax.text(
                    c,
                    r - 0.16,
                    f"{e['value']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=FONT["annot"],
                    color=col,
                    fontweight=weight,
                )
                ax.text(
                    c,
                    r + 0.22,
                    "—" if not np.isfinite(rv) else f"{rv:.0%}",
                    ha="center",
                    va="center",
                    fontsize=FONT["annot"],
                    color=col,
                    style="italic",
                )
        ax.set_xticks(range(len(DOMAINS)), [names.domain(d) for d in DOMAINS], fontsize=FONT["annot"])
        ax.set_yticks(
            range(len(labels)),
            [names.model(x) for x in labels] if i == 0 else [""] * len(labels),
            fontsize=FONT["annot"],
        )
        ax.set_title(f"{names.metric(metric)} ({names.direction_arrow(metric)})", fontsize=FONT["label"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    cx = L_ + len(metrics) * pw + (len(metrics) - 1) * GAP_ + 0.14
    cax = fig.add_axes((cx / TEXT_WIDTH_IN, (BOT + CAP) / figh, CBAR / TEXT_WIDTH_IN, grid_h / figh))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="min")
    cb.set_label("retention of the achievable gain", fontsize=FONT["annot"])
    cb.ax.tick_params(labelsize=FONT["annot"])
    cb.outline.set_visible(False)

    save_figure(fig, path, tag="transfer:panels", tight=False)


def transfer_gaps(cells, rows, metric: str, n_boot: int, seed: int):
    """Paired `LOO - all` on each row's held-out domain: the cost of never seeing that domain.

    Shared-episode resampling makes this far tighter than differencing two independent CIs, and it is
    the comparison the experiment exists to make.
    """
    stat = _statistic(metric)
    lower_better = names.lower_is_better(metric)
    gaps = []
    for label, held_out in ((r.label, r.held_out) for r in rows):
        if held_out is None:
            continue
        loo = cells.get(label, {}).get(held_out)
        ref = cells.get("all", {}).get(held_out)
        if not loo or not ref:
            continue
        ids_loo = {r["episode_id"] for r in loo}
        ids_ref = {r["episode_id"] for r in ref}
        if ids_loo != ids_ref:
            raise ValueError(
                f"{label}/{held_out}: episode sets differ from the `all` model "
                f"({len(ids_loo ^ ids_ref)} mismatched) — pairing requires identical episodes."
            )
        d, lo, hi = paired_macro_ci(
            {"loo": loo, "all": ref},
            stat,
            lambda m: m["loo"] - m["all"],
            n_boot=n_boot,
            seed=seed,
            stat_id=f"transfer_gap:{metric}:loo_minus_all",
        )
        # For a loss, positive delta = worse under transfer; for a score, negative = worse.
        worse = d > 0 if lower_better else d < 0
        significant = lo > 0 or hi < 0
        gaps.append(
            {
                "row": label,
                "domain": held_out,
                "delta": d,
                "lo": lo,
                "hi": hi,
                "direction": "worse" if worse else "better",
                "significant": bool(significant),
            }
        )
    return gaps


def render_markdown(matrix, rows, metric: str, gaps) -> str:
    cols = DOMAINS + ["all"]
    lines = [
        f"# Cross-domain transfer matrix — {names.metric(metric)} ({names.direction_arrow(metric)})",
        "",
        "Rows are models (by training mix); columns are test domains. `all` = **macro**-average over the",
        "three domains, with a stratified bootstrap in which the macro-average is the statistic.",
        "Both axes share the same order, so **the diagonal is the transfer cell** — that row's held-out",
        "domain, evaluated zero-shot (shown in **bold**). Everything off-diagonal is in-domain.",
        "",
        "| Model | " + " | ".join(_col_label(c) for c in cols) + " |",
        "|---" * (len(cols) + 1) + "|",
    ]
    for label in (r.label for r in rows):
        cells_txt = []
        for c in cols:
            e = matrix[label][c]
            t = _fmt(e["value"], e.get("lo"), e.get("hi"))
            cells_txt.append(f"**{t}**" if e.get("is_transfer") else t)
        lines.append(f"| {names.model(label)} | " + " | ".join(cells_txt) + " |")

    lines += ["", "## Transfer gaps (paired, shared-episode bootstrap)", ""]
    if not gaps:
        lines.append("_No transfer cells available yet._")
    else:
        lines += [
            "`delta` = LOO model − all-domain model, on the held-out domain. The pairing is what makes",
            "this tight: both models scored identical episodes, so shared variance cancels.",
            "",
            "| Held-out domain | Model | delta [95% CI] | Reading |",
            "|---|---|---|---|",
        ]
        for g in gaps:
            verdict = f"{g['direction']}{'' if g['significant'] else ' (CI includes 0)'}"
            lines.append(
                f"| {names.domain(g['domain'])} | {names.model(g['row'])} | "
                f"{_fmt(g['delta'], g['lo'], g['hi'])} | {verdict} |"
            )
    return "\n".join(lines) + "\n"


def render_heatmap(matrix, rows, metric: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = DOMAINS + ["all"]
    vals = np.array([[matrix[r.label][c]["value"] for c in cols] for r in rows], dtype=float)
    keys = [r.label for r in rows]  # raw keys index `matrix`
    labels = [names.model(k) for k in keys]  # display names are for tick text only
    lower_better = names.lower_is_better(metric)
    # Sequential map oriented so darker always means worse, whichever way the metric points.
    cmap = "viridis_r" if lower_better else "viridis"

    # Absolute inches at TEXT_WIDTH_IN rather than a per-cell figsize, so LaTeX places the figure at
    # `width=\textwidth` without shrinking the declared font sizes. Height follows from the row count so cells stay near-square.
    L_, R_, TOP_, BOT_, CBAR = 1.20, 0.62, 0.34, 0.50, 0.22
    grid_w = TEXT_WIDTH_IN - L_ - R_ - CBAR
    cell = grid_w / len(cols)
    grid_h = min(cell, 0.95) * len(rows)
    figh = TOP_ + grid_h + BOT_
    fig = new_figure(figh)
    ax = fig.add_axes((L_ / TEXT_WIDTH_IN, BOT_ / figh, grid_w / TEXT_WIDTH_IN, grid_h / figh))
    finite = vals[np.isfinite(vals)]
    im = ax.imshow(
        vals,
        cmap=cmap,
        vmin=(finite.min() if finite.size else 0),
        vmax=(finite.max() if finite.size else 1),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(
        range(len(cols)), [_col_label(c).replace(" (macro)", "\n(macro)") for c in cols], fontsize=FONT["tick"]
    )
    ax.set_yticks(range(len(keys)), labels, fontsize=FONT["tick"])
    ax.set_xlabel("Test domain", fontsize=FONT["label"])
    ax.set_ylabel("Model (training mix)", fontsize=FONT["label"])
    ax.set_title(
        f"Cross-domain transfer — {names.metric(metric)} ({names.direction_arrow(metric)})", fontsize=FONT["title"]
    )
    # No frame or tick marks: the cell grid already delimits the data, so spines are pure chrome.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i, key in enumerate(keys):
        for j, c in enumerate(cols):
            e = matrix[key][c]
            if not np.isfinite(e["value"]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=FONT["label"], color=GREY["note"])
                continue
            # Contrast against the cell fill rather than assuming a light background.
            rel = (e["value"] - im.get_clim()[0]) / max(im.get_clim()[1] - im.get_clim()[0], 1e-12)
            colour = "white" if (rel > 0.55) == lower_better else "black"
            if e.get("is_transfer"):
                # Value and marker are drawn separately so the annotation can stay small while the
                # number scales up; one combined two-line string would shrink both together.
                ax.text(
                    j,
                    i - 0.13,
                    f"{e['value']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=FONT["title"],
                    fontweight="bold",
                    color=colour,
                )
                ax.text(
                    j,
                    i + 0.24,
                    "transfer",
                    ha="center",
                    va="center",
                    fontsize=FONT["annot"],
                    style="italic",
                    color=colour,
                )
                # Exactly on the cell boundary, with clip_on=False. The stroke is centred on the
                # boundary, so for cells at the plot edge half of it lies outside the axes: clipping
                # would render those sides thinner than the interior ones. Disabling the clip keeps
                # every side identical without insetting.
                #
                # Do NOT inset instead: aspect="auto" means the data-unit -> pixel scale differs
                # between x and y, so one inset value becomes different pixel gaps horizontally and
                # vertically, and the box reads as off-centre. zorder is above the aggregate band so
                # that band cannot paint over the GWAS box.
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        ec="crimson",
                        lw=2.5,
                        zorder=6,
                        clip_on=False,
                    )
                )
            else:
                ax.text(j, i, f"{e['value']:.3f}", ha="center", va="center", fontsize=14, color=colour)

    # Set off the aggregate row/column (`all`, last on both axes) so the 3x3 per-domain block reads as
    # the data and the margins read as summaries rather than a fourth peer. The band width is in DATA
    # units (a fraction of a cell) rather than points, so it scales with the figure instead of shrinking
    # into a hairline the way an axhline `lw` does. Drawn over the cells, hence the zorder.
    # The band is one-sided: it encroaches ONLY on the aggregate row/column, never on the 3x3 block.
    # Centring it on the boundary would shave the neighbouring per-domain cell too, and the transfer
    # border (drawn at the true cell boundary) would then visibly overshoot the shrunken cell.
    gap = 0.15
    bg = dict(color=fig.get_facecolor(), zorder=5, lw=0)
    ax.add_patch(plt.Rectangle((-0.5, len(keys) - 1.5), len(cols), gap, **bg))
    ax.add_patch(plt.Rectangle((len(cols) - 1.5, -0.5), gap, len(keys), **bg))

    # Colorbar in its own absolutely-placed axes. `ax=ax` would steal width from the heatmap, which
    # breaks the cell geometry computed above; `tight_layout` would reflow everything and defeat the point.
    pos = ax.get_position()
    cax = fig.add_axes((pos.x1 + 0.10 / TEXT_WIDTH_IN, pos.y0, 0.10 / TEXT_WIDTH_IN, pos.height))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(names.metric(metric), fontsize=FONT["label"])
    cb.ax.tick_params(labelsize=FONT["tick"])
    cb.outline.set_visible(False)
    save_figure(fig, path, tag="transfer", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--testset",
        default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}",
        help="test set the results were produced against; used to derive the eval-tree subdirectory",
    )
    ap.add_argument(
        "--testset-revision",
        default=ep.DEFAULT_TESTSET_REVISION,
        help="required for a Hub test set: results from two publishes must not share a directory",
    )
    ap.add_argument("--eval-root", default=None, help="override the eval tree root (default IBIS_EVAL_ROOT)")
    ap.add_argument("--tag", default="primary", help="result tag: 'primary', or 'ctx<k>' for sensitivity")
    ap.add_argument("--metric", default="brier", help="brier|accuracy|ece|mce|auroc|auprc|brier_skill_score|smece")
    ap.add_argument("--rows", default=None, help="comma subset of row labels (default: all four)")
    ap.add_argument("--out-dir", default="outputs/transfer_matrix")
    ap.add_argument("--n-boot", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figure", action="store_true", help="skip the PDF (tables only)")
    ap.add_argument(
        "--panel-metrics",
        default="brier,reliability,resolution",
        help="comma list for the retention figure — one dot-and-whisker row per (held-out domain, metric). "
        "Any metric _statistic understands, so swapping to e.g. smece,auroc is a flag change. Empty disables.",
    )
    ap.add_argument(
        "--retention-matrix",
        action="store_true",
        help="also draw the 3x3-per-metric retention heatmap. Mostly a validity check: its off-diagonal "
        "is ~100% by construction, which confirms the scale but carries no result.",
    )
    ap.add_argument(
        "--combined-figure",
        action="store_true",
        help="also draw the compact Brier experiment matrix beside the transfer-retention plot",
    )
    args = ap.parse_args()

    rows = ROWS if not args.rows else [r for r in ROWS if r.label in {x.strip() for x in args.rows.split(",")}]
    if not rows:
        raise ValueError(f"--rows matched nothing; choose from {[r.label for r in ROWS]}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.eval_root) if args.eval_root else None
    slug = ep.testset_slug(args.testset, args.testset_revision)
    print(f"loading cells from {(root or ep.eval_root()) / slug} (tag={args.tag})")
    # The untrained base is loaded alongside, never displayed: retention needs it as the lower endpoint.
    cells = load_cells(slug, rows + [BASE_ROW], args.tag, root=root)
    have = sum(len(v) for v in cells.values())
    print(f"  {have}/{len(rows) * len(DOMAINS)} cells present")
    if not have:
        raise SystemExit("no result JSONs found — has the inference finished?")

    matrix = build(cells, rows, args.metric, args.n_boot, args.seed)
    gaps = transfer_gaps(cells, rows, args.metric, args.n_boot, args.seed) if "all" in cells else []

    md = out / f"transfer_matrix_{args.metric}.md"
    md.write_text(render_markdown(matrix, rows, args.metric, gaps))
    print(f"wrote {md}")

    csv = out / f"transfer_matrix_{args.metric}.csv"
    with csv.open("w") as f:
        f.write("model,eval_domain,value,ci_lo,ci_hi,n,is_transfer\n")
        for label in (r.label for r in rows):
            for c in DOMAINS + ["all"]:
                e = matrix[label][c]
                f.write(
                    f"{label},{c},{e['value']:.6f},{e.get('lo', float('nan')):.6f},"
                    f"{e.get('hi', float('nan')):.6f},{e['n']},{int(bool(e.get('is_transfer')))}\n"
                )
    print(f"wrote {csv}")

    # Side-by-side retention panels. Rebuilt per metric because each needs its own bootstrap, but the
    # cells are already loaded, so the cost is the bootstrap alone.
    panel_metrics = [m.strip() for m in args.panel_metrics.split(",") if m.strip()]
    if panel_metrics:
        loo_rows = [r for r in rows if r.held_out]
        per_metric = {}
        for m in panel_metrics:
            mm = build(cells, rows + [BASE_ROW], m, args.n_boot, args.seed)
            per_metric[m] = add_retention(mm, loo_rows)
        dots = {m: retention_ci(cells, loo_rows, m, args.n_boot, args.seed) for m in panel_metrics}
        losses = {m: transfer_losses(cells, loo_rows, m, args.n_boot, args.seed) for m in panel_metrics}

        required_metrics = {"brier", "reliability", "resolution"}
        required_rows = {"leave-out-tahoe", "leave-out-depmap", "leave-out-gwas", ALL_LABEL}
        if required_metrics.issubset(per_metric) and required_rows.issubset(row.label for row in rows):
            table = out / "transfer_full.tex"
            table.write_text(render_full_transfer_latex(per_metric, rows, dots, losses))
            print(f"wrote {table}")

        if not args.no_figure:
            # The dot plot is retained as a standalone diagnostic; the combined plot is the paper figure.
            render_retention_dots(dots, out / "transfer_retention")
            if args.combined_figure:
                missing = required_metrics - per_metric.keys()
                if missing:
                    raise ValueError(
                        f"--combined-figure requires {sorted(required_metrics)}; missing {sorted(missing)}"
                    )
                render_combined_transfer(per_metric["brier"], loo_rows, dots, out / "transfer_combined")
            if args.retention_matrix:
                render_retention_panels(per_metric, loo_rows, out / "transfer_retention_matrix")
        (out / "transfer_retention.json").write_text(
            json.dumps(
                {
                    "definition": "retention = (base - model) / (base - all-domain), per domain",
                    "base": BASE_ROW.eval_key,
                    "retention_ci": {
                        m: [{k: e[k] for k in ("held_out", "point", "lo", "hi")} for e in dots[m]]
                        for m in panel_metrics
                    },
                    "loss_vs_full_ci": {
                        m: [{k: e[k] for k in ("held_out", "point", "lo", "hi")} for e in losses[m]]
                        for m in panel_metrics
                    },
                    "metrics": {
                        m: {
                            r.label: {
                                d: {
                                    "value": per_metric[m][r.label][d]["value"],
                                    "retention": per_metric[m][r.label][d].get("retention"),
                                    "is_transfer": bool(per_metric[m][r.label][d].get("is_transfer")),
                                }
                                for d in DOMAINS
                            }
                            for r in loo_rows
                        }
                        for m in panel_metrics
                    },
                },
                indent=2,
            )
        )
        print(f"wrote {out / 'transfer_retention.json'}")

    # Full point-estimate suite per cell, so a later question does not need a re-run.
    suite = {}
    for label, held_out in ((r.label, r.held_out) for r in rows):
        suite[label] = {}
        for d, recs in cells.get(label, {}).items():
            pl = [(r["parsed_prob"], r["label"]) for r in recs if r.get("parsed_prob") is not None]
            if not pl:
                continue
            rep = compute_calibration(np.array([p for p, _ in pl]), np.array([float(y) for _, y in pl]))
            suite[label][d] = {"n": len(pl), "is_transfer": d == held_out, **rep.to_dict()}
            suite[label][d].pop("bins", None)  # keep the file readable; bins live in the figures
    (out / "cell_metrics.json").write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out / 'cell_metrics.json'}")

    if not args.no_figure:
        pdf = out / f"transfer_matrix_{args.metric}.pdf"
        render_heatmap(matrix, rows, args.metric, pdf)
        print(f"wrote {pdf} (+ .png)")

    print()
    print(render_markdown(matrix, rows, args.metric, gaps))


if __name__ == "__main__":
    main()
