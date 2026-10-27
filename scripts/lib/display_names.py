"""Canonical display names for domains, metrics, and model rows — one place, so every figure agrees.

The codebase keys everything on lowercase machine names (`tahoe`, `depmap`, `gwas`, `brier`), which is
right for filenames, JSON keys, and CLI flags. Those are *not* what belongs on an axis label: the
domains are proper nouns with specific capitalization (`DepMap`, `GWAS`) and the metrics have
conventional renderings (`Brier Score`, `AUROC`, `sMECE`). Left to each script, they drift — one figure
says "brier", the next "Brier", a third "Brier score".

Also carries each metric's **direction** (`lower_is_better`), because that is metric metadata rather
than a per-figure decision: it sets colormap orientation, the sign of a "good" delta, and whether an
axis annotation reads ↓ or ↑. A script that hardcodes its own loss list will eventually disagree with
another one.

Usage:
    import display_names as names
    names.domain("depmap")        # 'DepMap'
    names.metric("brier")         # 'Brier Score'
    names.model("leave-out-gwas") # 'Leave-out GWAS'
    names.lower_is_better("auroc")  # False

**Keep raw keys in data, display names in presentation.** CSV columns, JSON keys, and file paths should
stay lowercase machine names so downstream parsing keeps working; only the markdown/figure surface
should be prettified. Mixing the two is how a CSV ends up with a `DepMap` column that no script can
look up.

Unknown keys pass through unchanged rather than raising: a missing entry should show up as a visibly
odd label on a draft figure, not crash a long analysis run at the final step. Add the entry when you
see it.
"""

from __future__ import annotations

# Domains. `all` is the macro-aggregate pseudo-domain used as a matrix column / roster facet.
DOMAIN: dict[str, str] = {
    "tahoe": "Tahoe",
    "depmap": "DepMap",
    "gwas": "GWAS",
    "all": "All",
}

# Metrics. Keys match `scripts/lib/calibration_metrics.CalibrationReport` fields plus `accuracy`.
METRIC: dict[str, str] = {
    "brier": "Brier Score",
    "brier_skill_score": "Brier Skill Score",
    "accuracy": "Accuracy",
    "ece": "ECE",
    "mce": "MCE",
    "smece": "sMECE",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "reliability": "Reliability",
    "resolution": "Resolution",
    "uncertainty": "Uncertainty",
    "sharpness": "Sharpness",
    "base_rate": "Base Rate",
}

# Metrics where a SMALLER value is better (losses/calibration errors). Everything else is a score.
LOWER_IS_BETTER: frozenset[str] = frozenset({"brier", "ece", "mce", "smece", "reliability", "uncertainty"})


def domain(key: str) -> str:
    """Display name for a domain key (`'depmap'` -> `'DepMap'`); unknown keys pass through."""
    return DOMAIN.get(key, key)


def metric(key: str) -> str:
    """Display name for a metric key (`'brier'` -> `'Brier Score'`); unknown keys pass through."""
    return METRIC.get(key, key)


def model(key: str) -> str:
    """Display name for a model/row key.

    Handles the leave-one-out row labels compositionally, so a domain's capitalization is defined once:
    `'leave-out-gwas'` -> `'Leave-out GWAS'`. `'all'` becomes `'All domains'` — as a *model* it means
    "trained on everything", which is a different claim from the `All` macro *column*, and giving them
    the same label in one figure invites misreading.
    """
    if key == "all":
        return "All domains"
    if key.startswith("leave-out-"):
        return f"Leave-out {domain(key[len('leave-out-') :])}"
    return DOMAIN.get(key, key)


def lower_is_better(key: str) -> bool:
    """Whether a smaller value of this metric is better. Unknown metrics are treated as scores."""
    return key in LOWER_IS_BETTER


def direction_arrow(key: str) -> str:
    """`'↓'` / `'↑'` — the bare arrow, for axis labels and table headers.

    Callers add their own parentheses (`f"{metric(k)} ({direction_arrow(k)})"`), so the same helper works
    inside a LaTeX header, a markdown cell and a plot title without one of them ending up double-wrapped.
    The arrow alone is the convention in this literature; spelling out "lower is better" every time adds
    a line of text per column for something the glyph already says.
    """
    return "↓" if lower_is_better(key) else "↑"
