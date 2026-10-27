"""Cross-model calibration report for evidence deliberation.

Consumes one or more eval output JSONs (a top-level `results` list of records with
`parsed_prob` and `label`) and renders a single
self-contained HTML: a reliability-diagram overlay, a Brier/skill bar, and a
metrics table (Brier, BSS, ECE, MCE, AUROC, AUPRC, sharpness, resolution,
parse rate, truncation, mean output tokens). Trivial baselines (constant-0.5,
base-rate) are added as reference rows.

Usage:
  uv run python scripts/lib/render_calibration_report.py \\
      <model_a>.json <model_b>.json \\
      --out outputs/tahoe_calibration_report.html
"""

import argparse
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from calibration_metrics import compute_calibration

FONT_STACK = "din-2014, 'Helvetica Neue', Helvetica, Arial, sans-serif"
N_BINS = 15  # equal-mass reliability bins (~n/N_BINS forecasts each at n=2048)
matplotlib.rcParams.update(
    {
        "svg.fonttype": "none",  # keep text as selectable <text>, resolved by the page font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": "#1e2433",
        "axes.edgecolor": "#c4cad6",
        "axes.labelcolor": "#1e2433",
        "xtick.color": "#5c6370",
        "ytick.color": "#5c6370",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    }
)

CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: #fff; color: #1e2433; }
body { font-family: "din-2014","Helvetica Neue",Helvetica,Arial,sans-serif; font-size:15px; line-height:1.55; letter-spacing:-0.005em; }
article { max-width: 1100px; margin: 0 auto; padding: 48px 32px 96px; }
code { font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace; font-size:12.5px; padding:1px 5px; background:#f5f6f8; border-radius:4px; }
h1 { font-size:34px; font-weight:600; letter-spacing:-0.018em; margin:8px 0 8px; }
h2 { font-size:22px; font-weight:600; margin:48px 0 16px; }
.eyebrow { color:#5c6370; font-size:11px; letter-spacing:0.16em; text-transform:uppercase; margin:0 0 12px; }
.brand-rule { background:#ffcd00; border:0; height:4px; margin:0 0 24px; width:56px; }
.lede { font-size:17px; color:#5c6370; max-width:820px; margin:8px 0 28px; }
.table-wrap { border:1px solid #c4cad6; border-radius:8px; overflow:auto; margin:12px 0 16px; }
table { border-collapse:collapse; min-width:100%; }
th { background:#f5f6f8; color:#5c6370; font-size:11px; font-weight:600; letter-spacing:0.10em; text-transform:uppercase; padding:10px 12px; text-align:right; border-bottom:1px solid #c4cad6; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
td { padding:9px 12px; border-bottom:1px solid #e3e7ee; font-size:14px; text-align:right; white-space:nowrap; }
tr:last-child td { border-bottom:0; }
tr.baseline td { color:#5c6370; font-style:italic; background:#fafbfc; }
td.best { font-weight:600; background:#eafff4; }
figure { margin:16px 0 24px; }
.figrow { display:flex; gap:18px; align-items:flex-start; flex-wrap:wrap; }
.figrow > figure { flex:1 1 360px; min-width:0; margin:16px 0 24px; }
figure svg, figure img { width:100%; height:auto; border:1px solid #c4cad6; border-radius:10px; background:#fff; padding:10px; }
figcaption { color:#5c6370; font-size:12.5px; margin-top:8px; }
.note { color:#5c6370; font-size:12.5px; margin:8px 0 0; }
.footer { margin-top:64px; padding-top:20px; border-top:1px solid #c4cad6; color:#5c6370; font-size:12px; }
"""


_FIGDIR = None  # set by render() when --figdir is given; saves a standalone PDF per figure for LaTeX


def svg_fig(fig, name=None):
    """Render to inline SVG: vector (crisp), text stays selectable, page font.
    If a --figdir is configured and ``name`` is given, also save a standalone PDF."""
    if _FIGDIR is not None and name:
        fig.savefig(Path(_FIGDIR) / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg") :]  # drop xml/doctype preamble for inline embedding
    # make it scale to its container: drop the fixed pt width/height, keep viewBox
    svg = re.sub(r'(<svg[^>]*?)\s+width="[^"]*pt"\s+height="[^"]*pt"', r"\1", svg, count=1)
    # render every text element in the page font stack
    svg = re.sub(r"font-family\s*:\s*[^;}\"']+", "font-family:" + FONT_STACK, svg)
    return svg


# Okabe-Ito colorblind-safe categorical palette (assigned per model, in input order).
MODEL_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#9467BD"]

# Theming for a training progression vs an external comparison:
# the progression shares one hue that deepens with training step (latest darkest);
# comparison models get their own distinct hues.
SEQ_COLOR = "#0b5394"  # progression hue (deep blue)
COMP_COLORS = ["#e06c00", "#9b2fae", "#2e8b57"]  # external comparison models


def model_styles(valid, progression_k):
    """Return [(color, alpha)] aligned with ``valid``. If ``progression_k`` is set,
    the first k models are the training sequence (same hue, alpha ramps 0.4→1.0 so
    the latest is deepest); the rest get distinct comparison colors. Otherwise every
    model gets a distinct palette color."""
    if not progression_k:
        return [(MODEL_PALETTE[j % len(MODEL_PALETTE)], 1.0) for j in range(len(valid))]
    styles, ci = [], 0
    for j in range(len(valid)):
        if j < progression_k:
            alpha = 1.0 if progression_k <= 1 else 0.4 + 0.6 * j / (progression_k - 1)
            styles.append((SEQ_COLOR, alpha))
        else:
            styles.append((COMP_COLORS[ci % len(COMP_COLORS)], 1.0))
            ci += 1
    return styles


def grouped_metrics_fig(valid, styles):
    """One figure: a cluster of model bars per metric, sorted best->worst within
    each cluster, with per-model (color, alpha) from ``styles`` and bootstrap ±1 SE bars."""
    metrics = [
        ("brier", "Brier", "min"),
        ("ece", "ECE", "min"),
        ("auroc", "AUROC", "max"),
    ]
    style = {s["label"]: styles[i] for i, s in enumerate(valid)}
    fig, ax = plt.subplots(figsize=(11, 5))
    bw = 0.8 / max(1, len(valid))
    xticks, xlabels = [], []
    for gi, (attr, name, direction) in enumerate(metrics):
        items = [
            (s["label"], getattr(s["report"], attr), (s.get("boot") or {}).get(attr))
            for s in valid
            if getattr(s["report"], attr) is not None
        ]
        if not items:
            continue
        items.sort(key=lambda t: t[1], reverse=(direction == "max"))  # best -> worst (left -> right)
        k = len(items)
        for j, (lab, v, se) in enumerate(items):
            col, al = style[lab]
            x = gi + (j - (k - 1) / 2) * bw
            ax.bar(
                x,
                v,
                width=bw * 0.9,
                color=col,
                alpha=al,
                edgecolor="#1e2433",
                linewidth=0.6,
                yerr=se,
                capsize=3,
                error_kw={"ecolor": "#475569", "elinewidth": 1.0},
            )
            ax.text(x, v + (se or 0) + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#1e2433")
        xticks.append(gi)
        xlabels.append(f"{name} {'↓' if direction == 'min' else '↑'}")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("metric value")
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=style[s["label"]][0], alpha=style[s["label"]][1]) for s in valid]
    ax.legend(
        handles, [s["label"] for s in valid], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9
    )
    fig.tight_layout()
    return svg_fig(fig, "grouped_metrics")


def training_curve_fig(run_dir, max_step=None, color="#E69F00"):
    """Held-out eval reward vs training step (mean ±1 SE over eval episodes),
    read from {run_dir}/run_default/rollouts/step_*/eval_rollouts.jsonl."""
    import glob

    root = Path(run_dir) / "run_default" / "rollouts"
    paths = sorted(
        glob.glob(str(root / "step_*" / "eval_rollouts.jsonl")), key=lambda p: int(p.split("step_")[1].split("/")[0])
    )
    steps, means, ses = [], [], []
    for p in paths:
        step = int(p.split("step_")[1].split("/")[0])
        if max_step is not None and step > max_step:
            continue
        rew = [json.loads(line)["reward"] for line in open(p) if json.loads(line).get("reward") is not None]
        if not rew:
            continue
        a = np.asarray(rew, dtype=float)
        steps.append(step)
        means.append(float(a.mean()))
        ses.append(float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)
    if not steps:
        return ""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(
        steps,
        means,
        yerr=ses,
        marker="o",
        ms=4,
        color=color,
        ecolor="#94a3b8",
        elinewidth=1.0,
        capsize=2.5,
        linewidth=1.8,
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out eval reward")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return svg_fig(fig, "training_curve")


def _prob_on_truth_stats(path, n_boot=1000):
    """Mean probability the model assigned to the realized outcome (p if y==1 else
    1-p), with a bootstrap 95% CI, over the episodes in an eval JSON."""
    d = json.loads(Path(path).read_text())
    pt = [
        (r["parsed_prob"] if r["label"] == 1 else 1.0 - r["parsed_prob"])
        for r in d["results"]
        if r.get("parsed_prob") is not None
    ]
    if not pt:
        return None
    a = np.asarray(pt, dtype=float)
    n = len(a)
    rng = np.random.RandomState(0)
    boots = [a[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    return float(a.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n


def dose_response_fig(curves, styles, train_ctx=16):
    """curves: list of (label, [(n_stubs, mean, lo, hi), ...]); styles aligned.
    x = number of evidence stubs shown, y = mean P on the true outcome."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(0.5, ls="--", color="#a8b0bd", lw=1)
    ax.axvline(train_ctx, ls=":", color="#cbd2dd", lw=1.2)
    # anchor the label to the axes bottom in axis-fraction y (x stays in data coords) so it doesn't
    # depend on the not-yet-autoscaled y-limit — reading get_ylim() here (pre-plot) stretched the PDF.
    ax.text(
        train_ctx,
        0.02,
        " training\n context",
        transform=ax.get_xaxis_transform(),
        fontsize=7.5,
        color="#8b93a1",
        va="bottom",
    )
    for (label, pts), (col, al) in zip(curves, styles):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        yerr = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            ms=5,
            color=col,
            alpha=al,
            ecolor=col,
            elinewidth=1.0,
            capsize=3,
            linewidth=1.8,
            label=label,
        )
    ax.set_xlabel("number of evidence stubs shown")
    ax.set_ylabel("P assigned to the true outcome (mean ±95% CI)")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    return svg_fig(fig, "dose_response")


def model_label(data, override=None):
    if override:
        return override
    name = data.get("model", "?").split("/")[-1]
    tags = []
    if data.get("enable_thinking"):
        tags.append("think")
    if data.get("prompt_style") == "direct":
        tags.append("direct")
    if data.get("n_samples", 1) > 1:
        tags.append(f"bo{data['n_samples']}")
    return f"{name} ({', '.join(tags)})" if tags else name


def summarize(path, label_override=None):
    data = json.loads(Path(path).read_text())
    results = data["results"]
    probs, labels = [], []
    n_samples = n_trunc = 0
    tok_sum = 0
    for r in results:
        if r["parsed_prob"] is not None:
            probs.append(r["parsed_prob"])
            labels.append(r["label"])
        for s in r.get("samples", []):
            n_samples += 1
            n_trunc += int(s["is_truncated"])
            tok_sum += s["output_tokens"]
    # Equal-mass (quantile) bins: each bin holds ~n/B forecasts, so the reliability
    # diagram and ECE are robust to the model's prediction distribution (no sparse
    # singleton bins). smECE is the binning-free companion scalar.
    report = compute_calibration(probs, labels, n_bins=N_BINS, adaptive=True) if probs else None
    boot = {}
    if len(probs) > 1:
        P = np.asarray(probs, dtype=float)
        Y = np.asarray(labels, dtype=float)
        n = len(P)
        # bootstrap SE for the grouped-figure error bars (smece excluded: O(n^2), too slow here)
        acc = {k: [] for k in ("brier", "ece", "auroc", "auprc")}
        rng = np.random.RandomState(0)
        for _ in range(400):
            idx = rng.randint(0, n, n)
            rb = compute_calibration(P[idx].tolist(), Y[idx].tolist(), n_bins=N_BINS, adaptive=True, with_smece=False)
            for k in acc:
                v = getattr(rb, k)
                if v is not None:
                    acc[k].append(v)
        boot = {k: (float(np.std(v)) if v else None) for k, v in acc.items()}
    return {
        "label": model_label(data, label_override),
        "n_total": len(results),
        "n_parsed": len(probs),
        "parse_rate": len(probs) / len(results) if results else 0.0,
        "trunc_rate": n_trunc / n_samples if n_samples else 0.0,
        "mean_tokens": tok_sum / n_samples if n_samples else 0.0,
        "report": report,
        "boot": boot,
        "probs": probs,
        "labels": labels,
        "all_labels": [r["label"] for r in results],
    }


def baseline_rows(labels):
    """Constant-0.5 and base-rate reference forecasters on the same labels."""
    labels = list(labels)
    base = sum(labels) / len(labels)
    rows = []
    for name, p in [("constant p=0.5", 0.5), (f"base-rate p={base:.2f}", base)]:
        rep = compute_calibration([p] * len(labels), labels, n_bins=N_BINS, adaptive=True)
        rows.append({"label": name, "report": rep, "baseline": True})
    return rows


def reliability_curve_ci(probs, labels, n_bins=N_BINS, n_boot=500):
    """Equal-mass reliability curve with a bootstrap CI on each bin's empirical
    frequency. Returns per-bin (mean_pred, emp_freq, lo, hi, count) using fixed
    quantile edges; bins are wide where predictions are sparse (honest), and the
    CI widens where a bin holds few examples.

    `n_boot=0` skips the CI and returns NaN bounds -- for callers that only want the
    curve geometry (a figure that does not shade its bands pays a full bootstrap per
    model per domain otherwise). NaN rather than lo=hi=point: a zero-width interval
    would read as an impossibly precise estimate rather than as one never computed."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    idx = np.digitize(p, edges[1:-1], right=True)
    rng = np.random.RandomState(0)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        c = int(m.sum())
        if c == 0:
            continue
        yb = y[m]
        if n_boot <= 0:
            lo = hi = float("nan")
        elif c > 1:
            boot = [yb[rng.randint(0, c, c)].mean() for _ in range(n_boot)]
            lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        else:
            lo = hi = float(yb.mean())  # a genuinely singleton bin, not an uncomputed one
        out.append((float(p[m].mean()), float(yb.mean()), lo, hi, c))
    return out


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def render(
    summaries,
    out_path,
    title,
    run_dir=None,
    train_max_step=None,
    progression_k=None,
    dose_manifest=None,
    dose_progression=2,
    figdir=None,
):
    global _FIGDIR
    if figdir:
        _FIGDIR = figdir
        Path(figdir).mkdir(parents=True, exist_ok=True)
    valid = [s for s in summaries if s["report"] is not None]
    labels_for_baseline = valid[0]["all_labels"] if valid else []
    baselines = baseline_rows(labels_for_baseline) if labels_for_baseline else []
    styles = model_styles(valid, progression_k)

    # --- reliability overlay (equal-mass bins + bootstrap CI on empirical freq) ---
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1], [0, 1], "--", color="#a8b0bd", linewidth=1, label="perfect")
    for j, s in enumerate(valid):
        c, al = styles[j]
        if s.get("probs"):
            curve = reliability_curve_ci(s["probs"], s["labels"])
            px = [t[0] for t in curve]
            py = [t[1] for t in curve]
            yerr = [[t[1] - t[2] for t in curve], [t[3] - t[1] for t in curve]]
            ax.errorbar(
                px,
                py,
                yerr=yerr,
                marker="o",
                ms=4,
                color=c,
                ecolor=c,
                elinewidth=1.0,
                capsize=2.5,
                linewidth=1.8,
                alpha=al,
                label=f"{s['label']} (n={s['n_parsed']})",
            )
        else:  # manually-built summary without raw probs: fall back to point estimates
            bins = s["report"].bins
            if not bins:
                continue
            ax.plot(
                [b[0] for b in bins],
                [b[1] for b in bins],
                marker="o",
                color=c,
                alpha=al,
                linewidth=1.8,
                label=f"{s['label']} (n={s['n_parsed']})",
            )
    ax.set_xlabel("predicted P(supported)")
    ax.set_ylabel("empirical fraction supported")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="#c4cad6")
    fig.tight_layout()
    reliability_svg = svg_fig(fig, "reliability")

    grouped_svg = grouped_metrics_fig(valid, styles) if valid else ""

    # training curve in the deepest progression hue (ties it to the latest model)
    curve_color = SEQ_COLOR if progression_k else "#E69F00"
    training_svg = training_curve_fig(run_dir, train_max_step, color=curve_color) if run_dir else ""
    training_fig = (
        (
            "<figure>\n  " + training_svg + "\n"
            "  <figcaption>Held-out eval reward (mean ±1 SE over the 256-episode eval set) versus training "
            "step for the post-trained model's run. Reward = 0.9·Brier-reward + 0.1·format (higher is better).</figcaption>\n</figure>"
        )
        if training_svg
        else ""
    )

    dose_section = ""
    if dose_manifest:
        man = json.loads(Path(dose_manifest).read_text())  # {label: {n_stubs: json_path}}
        dose_labels = list(man.keys())
        dstyles = model_styles(dose_labels, dose_progression)
        curves = []
        for label in dose_labels:
            pts = []
            for ctx_str, path in sorted(man[label].items(), key=lambda kv: int(kv[0])):
                st = _prob_on_truth_stats(path)
                if st:
                    pts.append((int(ctx_str), st[0], st[1], st[2]))
            curves.append((label, pts))
        dose_svg = dose_response_fig(curves, dstyles)
        dose_section = (
            "<h2>Evidence sensitivity (dose-response)</h2>\n<figure>\n  " + dose_svg + "\n"
            "  <figcaption>Mean probability the model assigns to the realized outcome (p if supported, "
            "else 1−p) versus the number of evidence stubs shown, over a 512-episode held-out subset "
            "(±95% bootstrap CI); higher = stronger belief in the truth, 0.5 = chance. 0 stubs = prior only. "
            "The dotted line marks the training context size (16); the post-trained Qwen only trained at 16, "
            "so other stub counts probe generalization to unseen evidence amounts.</figcaption>\n</figure>"
        )

    # --- metrics table ---
    ordered = sorted(valid, key=lambda s: s["report"].brier) + baselines
    # Best across non-baseline models, per directional metric. Sharpness is
    # descriptive (no "better" direction) so it is never highlighted.
    metric_dir = {
        "brier": min,
        "brier_skill_score": max,
        "ece": min,
        "smece": min,
        "auroc": max,
        "auprc": max,
        "resolution": max,
    }
    best = {}
    for attr, pick in metric_dir.items():
        vals = [getattr(s["report"], attr) for s in valid if getattr(s["report"], attr) is not None]
        if vals:
            best[attr] = pick(vals)

    def cell(attr, val, is_base):
        hit = not is_base and attr in best and val is not None and abs(val - best[attr]) < 1e-9
        cls = " class='best'" if hit else ""
        return f"<td{cls}>{fmt(val)}</td>"

    rows_html = []
    for s in ordered:
        r = s["report"]
        is_base = s.get("baseline", False)
        extra = (
            ""
            if is_base
            else (
                f"<td>{s['n_parsed']}/{s['n_total']}</td>"
                f"<td>{s['parse_rate']:.0%}</td>"
                f"<td>{s['trunc_rate']:.0%}</td>"
                f"<td>{s['mean_tokens']:.0f}</td>"
            )
        )
        extra_base = "<td>—</td><td>—</td><td>—</td><td>—</td>" if is_base else ""
        rows_html.append(
            f"<tr class='{'baseline' if is_base else ''}'>"
            f"<td>{s['label']}</td>"
            + cell("brier", r.brier, is_base)
            + cell("brier_skill_score", r.brier_skill_score, is_base)
            + cell("ece", r.ece, is_base)
            + cell("smece", r.smece, is_base)
            + cell("auroc", r.auroc, is_base)
            + cell("auprc", r.auprc, is_base)
            + cell("resolution", r.resolution, is_base)
            + f"<td>{fmt(r.sharpness)}</td>"
            + f"{extra}{extra_base}"
            + "</tr>"
        )
    table_html = "\n".join(rows_html)

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://use.typekit.net/tfs6war.css">
<style>{CSS}</style></head>
<body><article>
<p class="eyebrow">EDB-Bio · calibration benchmark</p>
<hr class="brand-rule">
<h1>{title}</h1>
<h2>Models across metrics</h2>
<figure>
  {grouped_svg}
  <figcaption>One cluster per metric (↓ lower is better, ↑ higher is better). Within each cluster, bars are ordered best→worst left→right; color identifies the model (see legend). Whiskers are ±1 bootstrap standard error (400 resamples). Calibration metrics (Brier, ECE — equal-mass bins) and discrimination (AUROC) shown; BSS, AUPRC, the binning-free smECE, resolution and sharpness are in the table. For reference, a constant-p=0.5 forecaster scores Brier 0.25.</figcaption>
</figure>

<h2>Reliability and training dynamics</h2>
<div class="figrow">
  <figure>
    {reliability_svg}
    <figcaption>Reliability on the held-out slice, equal-mass bins (~n/15 forecasts each). Whiskers are bootstrap 95% CIs on each bin's empirical frequency, so thin/uncertain regions show wide bars. Points below the diagonal = overconfident in "supported"; above = underconfident.</figcaption>
  </figure>
  {training_fig}
</div>

<h2>Metrics</h2>
<div class="table-wrap"><table>
<thead><tr>
<th>Model</th><th>Brier ↓</th><th>BSS ↑</th><th>ECE ↓</th><th>smECE ↓</th><th>AUROC ↑</th><th>AUPRC ↑</th>
<th>Resol. ↑</th><th>Sharp.</th><th>Parsed</th><th>Parse%</th><th>Trunc%</th><th>Tok</th>
</tr></thead>
<tbody>{table_html}</tbody>
</table></div>
<p class="note">Green cell = best value in that column across the evaluated models (lower Brier / higher AUROC); baseline rows are excluded. It marks the relative leader only — it does <em>not</em> imply that model beats the constant-p=0.5 or base-rate forecaster (compare against the Brier 0.25 reference and the BSS column for that).</p>

{dose_section}

<p class="footer">Generated by render_calibration_report.py · metrics via calibration_metrics.compute_calibration (Murphy decomposition)</p>
</article></body></html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    print(f"Wrote {out_path} ({len(html_doc):,} bytes) — {len(valid)} models + {len(baselines)} baselines")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="eval_calibration.py output JSONs")
    p.add_argument("--out", default="outputs/tahoe_calibration_report.html")
    p.add_argument("--title", default="Are LLMs calibrated on scientific evidence?")
    p.add_argument("--labels", default=None, help="comma-separated label overrides, one per file")
    p.add_argument(
        "--colors",
        default=None,
        help="comma-separated hex palette, one per file (overrides the default MODEL_PALETTE)",
    )
    p.add_argument("--training-curve", default=None, help="RL run dir to plot eval-reward vs step at the bottom")
    p.add_argument("--training-curve-max-step", type=int, default=None, help="cap the training curve at this step")
    p.add_argument(
        "--progression",
        type=int,
        default=None,
        help="first K models form a training sequence (one hue, deepening); the rest are comparison models",
    )
    p.add_argument(
        "--dose-manifest", default=None, help="JSON {label: {n_stubs: eval_json_path}} to add a dose-response figure"
    )
    p.add_argument(
        "--dose-progression", type=int, default=2, help="first K dose models are the training sequence (deepening hue)"
    )
    p.add_argument(
        "--figdir", default=None, help="also export each figure as a standalone PDF into this dir (for LaTeX)"
    )
    args = p.parse_args()

    global MODEL_PALETTE
    if args.colors:
        MODEL_PALETTE = [c.strip() for c in args.colors.split(",")]

    overrides = args.labels.split(",") if args.labels else [None] * len(args.files)
    summaries = [summarize(f, lab) for f, lab in zip(args.files, overrides)]
    for s in summaries:
        r = s["report"]
        if r is None:
            print(f"  {s['label']}: NO parsed predictions ({s['n_total']} episodes)")
        else:
            print(
                f"  {s['label']}: Brier={r.brier:.4f} BSS={fmt(r.brier_skill_score)} "
                f"ECE={r.ece:.4f} AUROC={fmt(r.auroc)} parsed={s['n_parsed']}/{s['n_total']}"
            )
    render(
        summaries,
        args.out,
        args.title,
        run_dir=args.training_curve,
        train_max_step=args.training_curve_max_step,
        progression_k=args.progression,
        dose_manifest=args.dose_manifest,
        dose_progression=args.dose_progression,
        figdir=args.figdir,
    )


if __name__ == "__main__":
    main()
