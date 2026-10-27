"""Per-domain stub predicates for the conflict and position probes.

Both probes need three things a domain has to define for itself: which stub is the clearest
**supporting** measurement, which stubs are **neutral** enough to pad with, and (conflict only) what the
**mirror** of a supporting stub is. Tahoe's `verdict` field does not exist on DepMap or GWAS stubs, so
this module defines one predicate per domain, derived from the label rule each domain states in its own
system prompt.

**GWAS is deliberately absent from `mirror`.** The mirror has to be an equally compelling measurement of
the same experiment pointing the other way, and GWAS has no such thing at the evidence level: every stub
is a real published association, and a catalog holds no record asserting that a variant *fails* to
associate with a trait. Flipping `direction` on a catalog row fabricates a kind of record that does not
occur, so a model discounting it would be reacting to its implausibility rather than combining evidence
incoherently -- which is the quantity the probe reports. The paper already says this and already acts on
it: the weight-of-evidence analysis splits GWAS stubs by significance rather than supported/refuted.
Note this is a statement about *stubs*. GWAS *targets* are directional (supported iff p < 5e-8 with a
consistent direction), which is a different object and does not license a stub mirror.

The reflection point for DepMap is **zero, not the -0.5 dependency threshold**, because zero is the
no-effect point and the threshold is a decision boundary. Measured over the 1,488 supporting stubs in the
published test set, reflecting about zero renders every mirror as "knockout speeds growth -- suppressor",
genuinely refuting; reflecting about the threshold renders 1,326 of them as "mild fitness reduction",
which is weaker evidence rather than opposite evidence, and leaves 7 still reading "substantial fitness
cost", i.e. still supporting. Tahoe's existing mirror reflects log2FC about zero for the same reason.
"""

from __future__ import annotations

import copy

# DepMap's CRISPR card shows Chronos and P(dependency) side by side, each with a rendered call phrase, so
# a mirror that flips one and not the other produces a self-contradictory card ("suppressor" next to a
# 23% dependency probability, which is what 1-p would give). P(dependency) is a near-deterministic
# function of Chronos in DepMap's own model, so the mirror reads its P(dependency) off the empirical
# curve instead. Median P(dependency) per 0.25-wide Chronos bin over the 43,547 CRISPR stubs in the
# published primary + sensitivity configs; bins with fewer than 8 stubs dropped. Frozen here rather than
# refit at run time so the mirror is deterministic and reviewable.
_PDEP_BY_CHRONOS: list[tuple[float, float]] = [
    (-1.625, 1.0000),
    (-1.375, 0.9940),
    (-1.125, 0.9840),
    (-0.875, 0.9420),
    (-0.625, 0.7280),
    (-0.375, 0.3110),
    (-0.125, 0.0570),
    (+0.125, 0.0100),
    (+0.375, 0.0020),
    (+0.625, 0.0000),
    (+0.875, 0.0000),
]

# Top relevance tier per domain: the stub kind that bears most directly on the target, preferred when
# choosing which supporting stub to probe with. Falls back to any supporting stub.
TOP_TIER = {
    "tahoe": "same_drug_diff_cell",
    "depmap": "same_gene_same_lineage",
    "gwas": "same_gene_same_trait",
}

# The irrelevant-by-construction tier: an unrelated perturbation, gene or SNP-trait pair, present in
# every domain by principle P3. This is what makes the position probe portable at all -- GWAS has no
# statistically null stubs (every stub is a published association, median 0 per episode pass
# `is_neutral`), so relevance rather than significance has to supply the padding there.
CONTROL_TIER = {"tahoe": "negative_control", "depmap": "control", "gwas": "control"}

# Domains whose evidence admits a genuine equal-and-opposite mirror. See the module docstring.
MIRRORABLE = ("tahoe", "depmap")

DEPMAP_CHRONOS_THR = -0.5
DEPMAP_PDEP_THR = 0.5
GWAS_P_THR = 5e-8
TAHOE_LFC_THR = 0.5


def pdep_for_chronos(ch: float) -> float:
    """P(dependency) a real CRISPR stub at this Chronos value carries, by linear interpolation."""
    xs = [x for x, _ in _PDEP_BY_CHRONOS]
    ys = [y for _, y in _PDEP_BY_CHRONOS]
    if ch <= xs[0]:
        return ys[0]
    if ch >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= ch <= xs[i + 1]:
            t = (ch - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _target_is_up(row: dict) -> bool:
    return str(row.get("target_direction", "")).lower().startswith("up")


def is_supporting(domain: str, stub: dict, row: dict) -> bool:
    """Does this stub, on its own, point toward the target hypothesis being supported?

    One clause per domain, matching the rule that domain's system prompt states to the model.
    """
    y = stub.get("y") or {}
    if domain == "tahoe":
        # Significant differential expression whose sign matches the claimed direction.
        p, lfc = y.get("padj"), y.get("log2FoldChange")
        if p is None or lfc is None:
            return False
        return p < 0.05 and abs(lfc) > TAHOE_LFC_THR and ((lfc > 0) == _target_is_up(row))
    if domain == "depmap":
        # "Chronos gene-effect <= -0.5 AND P(dependency) >= 0.5", the prompt's own wording.
        ch, pdep = y.get("chronos"), y.get("p_dependency")
        if ch is None or pdep is None:
            return False
        return ch <= DEPMAP_CHRONOS_THR and pdep >= DEPMAP_PDEP_THR
    if domain == "gwas":
        # Genome-wide significant with an effect direction consistent with the target's.
        p, d = y.get("p_value"), y.get("direction")
        if p is None or d is None:
            return False
        td = row.get("target_direction")
        consistent = True if td is None else (float(d) > 0) == str(td).lower().startswith(("up", "pos", "+"))
        return p < GWAS_P_THR and consistent
    raise ValueError(f"unknown domain {domain!r}")


def is_neutral(domain: str, stub: dict) -> bool:
    """Is this stub uninformative enough to pad a context with?

    The position probe needs padding that barely moves the belief, so that a late-position probe stub is
    not competing with a saturated prior. Deliberately stricter than "not supporting": a refuting stub is
    not supporting either, and padding with refuting evidence would drag the belief down instead of
    holding it near the prior.
    """
    y = stub.get("y") or {}
    if domain == "tahoe":
        return stub.get("verdict") == "inconclusive"
    if domain == "depmap":
        ch, pdep = y.get("chronos"), y.get("p_dependency")
        if ch is None or pdep is None:
            # Molecular and pharmacology modalities carry no Chronos call at all, so they cannot assert a
            # dependency either way. Usable as padding.
            return stub.get("modality") in ("expression", "copy_number", "mutation", "pharmacology")
        return -0.2 <= ch <= 0.25 and pdep < 0.1
    if domain == "gwas":
        p = y.get("p_value")
        return p is not None and p > 1e-5
    raise ValueError(f"unknown domain {domain!r}")


def is_padding(domain: str, stub: dict) -> bool:
    """Can this stub pad a position-probe context without moving the belief much?

    Either statistically uninformative (`is_neutral`) or irrelevant by construction (the control tier).
    The union is what makes the probe portable: Tahoe has ~30 usable stubs per episode either way, but
    DepMap has ~11 and GWAS ~12 only once the control tier counts, and GWAS has essentially none without
    it. Measured over the TEST split, the union leaves >1,000 eligible episodes per domain at a pool of
    ten, against 1,191 / 14 / 472 at a pool of sixteen -- which is why the probe runs at ten.
    """
    return is_neutral(domain, stub) or stub.get("context_type") == CONTROL_TIER.get(domain)


def support_rank(domain: str, stub: dict) -> tuple:
    """Sort key picking the *clearest* supporting stub; lower is better. Relevance tier first, then
    statistical strength, so the probe stub is the one a reader would find most compelling."""
    y = stub.get("y") or {}
    tier = 0 if stub.get("context_type") == TOP_TIER.get(domain) else 1
    if domain == "tahoe":
        return (tier, y.get("padj", 1.0), -abs(y.get("log2FoldChange", 0.0)))
    if domain == "depmap":
        return (tier, y.get("chronos", 0.0), -float(y.get("p_dependency", 0.0)))
    if domain == "gwas":
        return (tier, y.get("p_value", 1.0), -abs(float(y.get("or_beta") or 0.0)))
    raise ValueError(f"unknown domain {domain!r}")


def mirror(domain: str, stub: dict) -> dict:
    """The equal-and-opposite twin of a supporting stub: same experiment, same magnitude, other side.

    Every field the model is shown must move together, or the card contradicts itself and the probe
    measures the model's reaction to an incoherent record rather than to conflicting evidence.
    """
    if domain not in MIRRORABLE:
        raise ValueError(f"{domain} has no refuting stub category, so no mirror exists; see the module docstring")
    m = copy.deepcopy(stub)
    y = m["y"]
    if domain == "tahoe":
        # padj is two-sided, so it is unchanged; the rendered up/down label derives from the sign.
        y["log2FoldChange"] = -y["log2FoldChange"]
        if y.get("stat") is not None:
            y["stat"] = -y["stat"]
        m["verdict"] = "refuted"
    else:
        ch = -float(y["chronos"])
        y["chronos"] = ch
        y["p_dependency"] = round(pdep_for_chronos(ch), 4)
    return m
