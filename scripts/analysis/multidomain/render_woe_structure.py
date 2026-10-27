"""Weight-of-evidence structure on the published multi-domain TEST set.

Does a model weight evidence the way the domain says it should — more movement from a statistically strong
stub than a null one, more from a relevant tier than a marginal one, and in the *right direction*?
Conventions: **decibans** (`10/ln 10`), `EPS = 1e-3` logit clipping, Spearman ρ against tier rank,
episode-clustered bootstrap CIs, and each domain's own tier hierarchy and verdict rule.

**Only the `sensitivity` config can produce this.** A per-stub weight of evidence is a *difference between
two forecasts on the same episode*,

    WoE_j = [logit(p_j) − logit(p_{j−1})] × 10/ln 10       (decibans, j = 1..16)

so it needs that episode scored at both j−1 and j. `primary` scores each episode at one k, so it cannot
express a single increment. `--tag` (not `--testset`) selects the config — see `render_dose_response.py`.

**Attribution rests on the pool being prefix-nested**: the k-stub context is the first k stubs of the
16-stub context. That is asserted, not assumed — if it broke, every increment would be attributed to the
wrong stub. Given nesting, the stub added at step j is `context_pool[j−1]`.

**One figure per question:**

| Output | Shows |
|---|---|
| `woe_by_tier` | mean \\|WoE\\| vs relevance tier, per domain, with Spearman ρ |
| `woe_by_verdict` | \\|WoE\\| distribution by stub verdict — **violins**, per domain |
| `woe_directional` | mean **signed** WoE by verdict: does support push up and refutation push down |
| `woe_by_position` | mean \\|WoE\\| vs stub position — primacy / saturation |

Domains keep their own axes because their taxonomies genuinely differ in *kind*, not just in length:
tahoe verdicts are `supported/refuted/inconclusive/aggregate`, depmap's are derived per modality, and GWAS
is **significance-only** (`significant`/`sub_threshold`). GWAS is therefore **excluded from the directional
figure**: a GWAS stub's `direction` is its own allele's sign, not consistency with the target hypothesis,
so the domain has no genuine refuting category and a signed-by-verdict panel would be meaningless — the
same reason `render_gwas_woe.py` omits it.

`|WoE|` **saturates** near 60 decibans (`|logit(EPS)| × 2 × 10/ln 10`); a model answering an exact 0 or 1
hits it. The clamp rate is reported per model, because a model that saturates often has a `|WoE|` mean that
is a property of `EPS` rather than of the model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = next(p for p in _HERE.parents if p.name == "scripts").parent
sys.path.insert(0, str(_ROOT / "scripts" / "lib"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "envs" / "ibis_env_utils"))

import display_names as dn  # noqa: E402
import eval_paths  # noqa: E402
from figure_io import GREY, TEXT_WIDTH_IN, save_figure  # noqa: E402
from ibis_env_utils import Split, load_dataset_split  # noqa: E402
from macro_bootstrap import macro_ci, macro_point  # noqa: E402
from multidomain_cells import DOMAINS, cell_paths_for_tag  # noqa: E402
from roster_palette import ORDER, ROSTER, plot_color  # noqa: E402

DECIBAN = 10.0 / np.log(10.0)
EPS = 1e-3
CLAMP_DB = 29.0  # |WoE| at/above this is treated as saturated
# Percentile of pooled |WoE| that sets the directional violins' y-limit (rounded up to a multiple of 5).
# Not a hand-picked number: it tracks the data if the roster changes. 97 keeps every model's mean and
# interquartile mass on-panel while cutting the saturation tail that would otherwise set the axis.
CLIP_PCT = 97.0
GW_SIG = 5e-8  # genome-wide significance bar
INK, GRAY = GREY["ink"], GREY["note"]

# Curated relevance hierarchies, most -> least relevant.
HIER: dict[str, list[str]] = {
    "gwas": [
        "same_gene_same_trait",
        "same_gene_related_trait",
        "same_snp_related_trait",
        "same_snp_unrelated_trait",
        "pathway_gene_same_trait",
        "control",
    ],
    "depmap": [
        "same_gene_same_lineage",
        "same_gene_other_lineage",
        "coessential_pathway",
        "pharmacology",
        "molecular",
        "control",
    ],
    "tahoe": [
        "same_drug_diff_cell",
        "same_moa_evidence",
        "diff_drug_same_cell",
        "same_drug_cell_aggregate",
        "negative_control",
    ],
}
# The published depmap set keeps molecular_mutation / molecular_cn / molecular_expression separate; they
# are collapsed into ONE "molecular" tier. Collapse rather than expand: ρ is computed against tier rank,
# and any internal order among the three would be invented here.
TIER_ALIAS: dict[str, str] = {
    "molecular_mutation": "molecular",
    "molecular_cn": "molecular",
    "molecular_expression": "molecular",
}
CONTROL: dict[str, str] = {"gwas": "control", "depmap": "control", "tahoe": "negative_control"}

# Single-line short names for the staircase's rotated tick labels. TIER_PRETTY's embedded newlines are
# right for a wide horizontal axis and unusable here: six ticks share a 1.6in panel, so a horizontal
# multi-line label is far wider than its slot and neighbouring labels overprint into unreadable mush.
# Rotated labels have to be one line, and short enough that the rotated extent fits the bottom margin.
TIER_SHORT = {
    "same_gene_same_trait": "same gene, same trait",
    "same_gene_related_trait": "same gene, related trait",
    "same_snp_related_trait": "same SNP, related trait",
    "same_snp_unrelated_trait": "same SNP, unrelated trait",
    "pathway_gene_same_trait": "pathway gene, same trait",
    "same_gene_same_lineage": "same gene, same lineage",
    "same_gene_other_lineage": "same gene, other lineage",
    "coessential_pathway": "co-essential pathway",
    "pharmacology": "drug vs. target/pathway",
    "molecular": "molecular state",
    "same_drug_diff_cell": "same drug, other cell",
    "same_moa_evidence": "same mechanism",
    "diff_drug_same_cell": "other drug, same cell",
    "same_drug_cell_aggregate": "aggregate summary",
    "control": "unrelated control",
    "negative_control": "unrelated control",
}
TIER_PRETTY = {
    "same_gene_same_trait": "same gene,\nsame trait",
    "same_gene_related_trait": "same gene,\nrelated trait",
    "same_snp_related_trait": "same SNP,\nrelated trait",
    "same_snp_unrelated_trait": "same SNP,\nunrelated trait",
    "pathway_gene_same_trait": "pathway-neighbor\ngene, same trait",
    "same_gene_same_lineage": "same gene,\nsame-lineage cell",
    "same_gene_other_lineage": "same gene,\nother lineage",
    "coessential_pathway": "co-essential\npathway partner",
    "pharmacology": "drug vs.\ntarget/pathway",
    "molecular": "target gene's\nmolecular state",
    "same_drug_diff_cell": "same drug,\nother cell line",
    "same_moa_evidence": "different drug,\nsame mechanism",
    "diff_drug_same_cell": "different drug &\nmechanism, same cell",
    "same_drug_cell_aggregate": "aggregate summary\n(target drug x cell)",
    "control": "unrelated\ncontrol",
    "negative_control": "unrelated:\ndifferent drug,\nmechanism & cell",
}

# Verdict vocabularies, per domain, from the sources. GWAS is significance-only.
VERDS: dict[str, list[str]] = {
    "tahoe": ["supported", "refuted", "inconclusive", "aggregate"],
    "depmap": ["supported", "refuted", "inconclusive"],
    "gwas": ["significant", "sub_threshold"],
}
VERD_PRETTY = {"significant": "genome-wide\nsignificant", "sub_threshold": "sub-threshold"}
# Domains with a genuine refuting category, hence a meaningful signed-WoE figure (GWAS has none).
DIRECTIONAL_DOMAINS = ("tahoe", "depmap")

# Verdicts shown in the directional VIOLIN figure, per domain.
#
# GWAS is INCLUDED: drawing its two categories with no
# refuting slot states the domain's structure in the figure, where omitting the domain hid it. A
# significant hit at another locus corroborates rather than contradicts the target, so GWAS has no
# refuting category to draw -- the missing third violin IS the caveat, and it is labelled in-panel.
#
# Tahoe's `aggregate` is dropped HERE ONLY. It is not a strength level: it is a distinct evidence KIND (a
# pooled summary over the target drug x cell, 3.3% of stubs, the only Tahoe stubs carrying no verdict at
# all), so on an ordered strength axis it does not belong. It stays in the tier hierarchy, where it is a
# legitimate tier -- and where it is worth a sentence, being the single largest cross-model disagreement
# anywhere in this analysis (frontier models read it as supporting, base-direct as refuting).
DIRECTIONAL_VERDS: dict[str, list[str]] = {
    "tahoe": ["supported", "refuted", "inconclusive"],
    "depmap": ["supported", "refuted", "inconclusive"],
    "gwas": ["significant", "sub_threshold"],
}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")

    def rank(x):
        order = x.argsort()
        r = np.empty(len(x), float)
        r[order] = np.arange(1, len(x) + 1)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)  # average ties
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]

    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def stub_verdict(domain: str, stub: dict) -> str:
    """This domain's verdict rule.

    tahoe: the stub carries `verdict` outright (`aggregate` when absent — an aggregate-summary stub).
    depmap: derived per modality.
    gwas: significance only.
    """
    y = stub.get("y") or {}
    if domain == "tahoe":
        return stub.get("verdict") or "aggregate"
    if domain == "gwas":
        p = y.get("p_value")
        return "significant" if (p is not None and p < GW_SIG) else "sub_threshold"
    mod = stub.get("modality")
    if mod == "crispr":
        ch, pd_ = y.get("chronos"), y.get("p_dependency")
        if ch is not None and pd_ is not None:
            if pd_ >= 0.5 and ch <= -0.5:
                return "supported"
            if ch > 0.25:
                return "refuted"
        return "inconclusive"
    if mod == "expression":
        return "supported" if (y.get("lineage_z") or 0) >= 1.5 else "inconclusive"
    if mod == "copy_number":
        return "supported" if y.get("call") == "amplified" else "inconclusive"
    if mod == "mutation":
        return "supported" if y.get("status") == "mutated" else "inconclusive"
    if mod == "pharmacology":
        return "supported" if y.get("sensitive") else "inconclusive"
    return "inconclusive"


def stub_meta(testset: str, revision: str | None = None) -> dict[tuple[str, str], list[dict]]:
    """{(domain, episode_id): [{tier, verdict, signed_dir} per position]}, asserting prefix-nesting.

    `revision` MUST be the same revision the cells were scored against. Per-stub WoE is attributed by
    matching `(domain, episode_id)` between these stub pools and the eval cells, so reading the pools from
    a moving `main` while the cells come from a pinned revision silently attributes increments to the
    wrong stubs where episodes changed, and silently drops them where episode ids do not exist in both.
    Neither failure raises. `main()` passes `--revision` here for exactly that reason.
    """
    out: dict[tuple[str, str], list[dict]] = {}
    for d in DOMAINS:
        df = load_dataset_split(testset.format(domain=d), Split.TEST, revision=revision)
        widest = int(df["k"].max())
        # `testset` must name the DENSE (sensitivity) config. Unlike the eval cells -- which share one
        # directory per model because the slug keys on repo + revision, not config -- this reads the actual
        # parquet, and primary and sensitivity are different files with different shapes. Primary holds one
        # k per episode, so most episodes have no row at the widest k and the attribution below is
        # impossible. Checked once, up front, because the failure was otherwise an `iloc[0]` IndexError
        # from deep inside a groupby with nothing pointing at the cause.
        n_at_widest = int((df["k"] == widest).sum())
        n_eps = df["episode_id"].nunique()
        if n_at_widest < n_eps:
            raise SystemExit(
                f"{d}: only {n_at_widest}/{n_eps} episodes have a row at the widest k={widest}, so this is "
                f"not a dense sweep. Per-stub WoE needs the FULL stub pool per episode.\n"
                f"  --testset was {testset!r}\n"
                f"  Point it at the `sensitivity/` config, not `primary/` — primary carries one k per "
                f"episode by design."
            )
        for eid, g in df.groupby("episode_id"):
            g = g.sort_values("k")
            full = list(g[g.k == widest].iloc[0]["context_pool"])
            ftypes = [s.get("context_type") for s in full]
            for r in g.itertuples():
                if [s.get("context_type") for s in r.context_pool] != ftypes[: int(r.k)]:
                    raise ValueError(
                        f"{d}/{eid}: context_pool at k={r.k} is not a prefix of the k={widest} pool, so a "
                        f"per-stub WoE cannot be attributed to a stub. Rebuild the test set."
                    )
            meta = []
            for s in full:
                t = s.get("context_type")
                v = stub_verdict(d, s)
                meta.append(
                    {
                        "tier": TIER_ALIAS.get(t, t),
                        "verdict": v,
                        "signed_dir": {"supported": 1, "refuted": -1}.get(v, 0),
                    }
                )
            out[(d, str(eid))] = meta
    return out


def woe_records(slug, eval_key, tag, root, meta) -> tuple[list[dict], dict]:
    """[{domain, episode_id, position, tier, verdict, woe, abswoe}] + diagnostics, for one model."""
    p: dict[tuple[str, str], dict[int, float]] = {}
    for d in DOMAINS:
        for path in cell_paths_for_tag(slug, eval_key, d, tag, root=root):
            k = int(path.stem.rsplit("_ctx", 1)[1])
            for r in json.loads(path.read_text())["results"]:
                if r.get("parsed_prob") is not None:
                    p.setdefault((d, str(r["episode_id"])), {})[k] = float(r["parsed_prob"])
    # Both `continue`s below are COUNTED, not merely skipped: uncounted, a mismatch between the stub pools
    # and the scored cells is invisible, because the figure renders from whatever did line up and looks
    # complete. They are not equally diagnostic, though:
    #
    # - `n_no_meta` (episode scored, no stub pool at all) is the RELIABLE skew signal. It is zero whenever
    #   the pools and the cells describe the same episodes, so non-zero means they do not.
    # - `n_short_pool` (a k past the end of its pool) is NOT a skew signal, and must not be reported as
    #   one. Some episodes genuinely hold fewer stubs than the sweep's widest k -- one in Tahoe and one in
    #   DepMap at revision v1.0.0 -- and a pool that shrank across revisions is indistinguishable from a
    #   pool that was always short. Counted for the record, never warned on.
    recs, n_gap, n_no_meta, n_short_pool = [], 0, 0, 0
    for key, byk in p.items():
        ms = meta.get(key)
        if ms is None:
            n_no_meta += 1
            continue
        for k in sorted(byk):
            if k == 0 or (k - 1) not in byk:
                n_gap += k > 0  # one endpoint unparsed -> increment not computable
                continue
            if k > len(ms):
                n_short_pool += 1
                continue
            w = float((logit(byk[k]) - logit(byk[k - 1])) * DECIBAN)
            recs.append(
                {"domain": key[0], "episode_id": key[1], "position": k, **ms[k - 1], "woe": w, "abswoe": abs(w)}
            )
    # Episode-normalized |WoE|: each stub divided by the mean |WoE| of its OWN episode, so 1.0 reads as
    # "an average stub for this episode". Raw decibans are not comparable across models -- base-direct
    # moves 5-8 db where direct RL moves 0.4-1.2 -- so anything that pools or averages over models (the
    # derived tier order, the staircase's shared y-axis) must run on this scale or it silently becomes a
    # report about whichever model is loudest rather than about the models collectively.
    ep_sum: dict[tuple[str, str], list[float]] = {}
    for r in recs:
        ep_sum.setdefault((r["domain"], r["episode_id"]), []).append(r["abswoe"])
    ep_mean = {k: float(np.mean(v)) for k, v in ep_sum.items()}
    for r in recs:
        m = ep_mean[(r["domain"], r["episode_id"])]
        r["abswoe_norm"] = float(r["abswoe"] / m) if m > 0 else float("nan")

    clamp = float(np.mean([r["abswoe"] >= CLAMP_DB for r in recs])) if recs else float("nan")
    return recs, {
        "n_increments": len(recs),
        "n_skipped_unparsed_endpoint": n_gap,
        "n_episodes_without_stub_meta": n_no_meta,  # revision skew; see the comment above
        "n_skipped_short_stub_pool": n_short_pool,  # data property, NOT skew
        "n_episodes_scored": len(p),
        "clamp_rate": clamp,
    }


def _mean(field: str):
    def f(rows):
        v = [r[field] for r in rows]
        return float(np.mean(v)) if v else float("nan")

    return f


def gap_stat(rows) -> float:
    """mean |WoE| on relevant stubs − mean |WoE| on control stubs, within one domain."""
    rel = [r["abswoe"] for r in rows if r["tier"] != CONTROL.get(r["domain"])]
    ctl = [r["abswoe"] for r in rows if r["tier"] == CONTROL.get(r["domain"])]
    if not rel or not ctl:
        return float("nan")
    return float(np.mean(rel) - np.mean(ctl))


def _style(ax):
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)


def _save(fig, base: Path, name: str, tight: bool = True) -> None:
    """Write PDF + PNG via the shared saver. `tight=False` preserves the authored size exactly, which the
    two paper panels need because they are laid out in absolute inches to match \\textwidth."""
    save_figure(fig, base / name, tag=f"woe:{name}", tight=tight, facecolor="white", close=False)


def _panels(n: int, w: float, h: float):
    import matplotlib.pyplot as plt

    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    return fig, axes.ravel()


def fig_by_tier(series, keys, out: Path) -> None:
    """mean |WoE| vs relevance tier, one panel per domain, ρ in the legend. Model curves are SOLID."""
    fig, flat = _panels(len(DOMAINS), 6.0, 4.9)
    for ax, d in zip(flat, DOMAINS):
        levels = [t for t in HIER[d] if any(t in series[k]["tier_means"].get(d, {}) for k in keys)]
        for key in keys:
            s, e = series[key], ROSTER[key]
            ys = [s["tier_means"].get(d, {}).get(t, np.nan) for t in levels]
            ax.plot(
                range(len(levels)),
                ys,
                color=plot_color(e),
                lw=2.2,
                marker="o",
                ms=4,
                label=f"{e.label}  ρ={s['rho'].get(d, float('nan')):+.2f}",
            )
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([TIER_PRETTY.get(t, t) for t in levels], fontsize=7.5, rotation=30, ha="right")
        ax.set_title(f"{dn.domain(d)}", fontsize=11)
        ax.set_xlabel("evidence tier   (highest → lowest relevance)", fontsize=9, color=GRAY)
        ax.set_ylabel("mean |WoE| per stub  (decibans)", fontsize=9, color=GRAY)
        ax.legend(fontsize=6.6, frameon=False)
        _style(ax)
    fig.suptitle("Weight of evidence by relevance tier — multi-domain TEST set (sensitivity sweep)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out, "woe_by_tier")


def _append_order_table(out_dir: Path, orders: dict[str, list[str]]) -> None:
    """Append the derived tier order to woe_structure.md, next to the Spearman table.

    The two belong together and answer different questions. ρ scores each model against the CURATED
    hierarchy — "does this model agree with the domain expert?". The derived order says what the models
    collectively respond to. Where they disagree, that disagreement is the result, and it is only
    interpretable if both orderings are on the page.
    """
    lines = ["", "### Derived vs curated tier order", ""]
    for d, levels in orders.items():
        cur = HIER[d]
        moved = [f"{t} ({cur.index(t) + 1}→{i + 1})" for i, t in enumerate(levels) if cur.index(t) != i]
        lines.append(f"**{dn.domain(d)}** — models rank: " + " > ".join(levels))
        lines.append("")
        lines.append("  curated: " + " > ".join(cur))
        lines.append("")
        lines.append(f"  moved: {', '.join(moved) if moved else 'nothing — the orders agree'}")
        lines.append("")
    lines += [
        "Derived order = tiers sorted by cross-model mean episode-normalized |WoE| (each stub divided by "
        "its own episode's mean |WoE|, so a louder model cannot dominate the ordering). The ρ column above "
        "stays scored against the **curated** hierarchy: ρ answers 'does this model agree with the domain "
        "expert', the derived order answers 'what do the models collectively find compelling'.",
        "",
    ]
    p = out_dir / "woe_structure.md"
    p.write_text(p.read_text() + "\n".join(lines) + "\n")


def tier_profile(recs: list[dict], domain: str) -> dict[str, float]:
    """{tier: mean episode-normalized |WoE|} for one model in one domain."""
    by_t: dict[str, list[float]] = {}
    for r in recs:
        if r["domain"] == domain and not np.isnan(r["abswoe_norm"]):
            by_t.setdefault(r["tier"], []).append(r["abswoe_norm"])
    return {t: float(np.mean(v)) for t, v in by_t.items()}


def empirical_order(profiles: dict[str, dict[str, float]], domain: str) -> list[str]:
    """Tiers ordered by cross-model mean normalized |WoE|, most-moved first.

    The curated hierarchies encode what a domain expert judges relevant. This orders the same tiers by
    what the models actually respond to, which is a different claim and the one a staircase can show.
    Plotted against the curated order the profiles zig-zag and the shape is unreadable, because several
    tiers are consistently mis-placed relative to how models treat them; ordered by the data the staircase
    is close to monotone and the DEPARTURE from the curated order becomes the finding. That departure is
    quantified separately by the Spearman table, which stays scored against the curated hierarchy -- the
    two together are the comparison ("what models find compelling" vs "what the domain says is relevant").

    Every model contributes one term to the mean, on the normalized scale, so a louder model cannot
    dominate the ordering.
    """
    tiers = HIER[domain]
    mean_by_tier = {}
    for t in tiers:
        vals = [p[t] for p in profiles.values() if t in p]
        if vals:
            mean_by_tier[t] = float(np.mean(vals))
    return sorted(mean_by_tier, key=lambda t: -mean_by_tier[t])


def fig_staircase(recs_by_model, keys, out: Path) -> dict[str, list[str]]:
    """Evidence hierarchy as lollipops, tiers in the CURATED (expert) order.

    One panel per domain; y is **normalized WoE** -- each stub's |WoE| over the mean |WoE| of its own
    episode -- on a log2 axis labelled 0.5x / 1x / 2x, with a stem from the episode mean to each model's
    dot. Models sit in fixed slots inside each tier column.

    Step lines in the curated order zig-zag into an unreadable tangle; unconnected lollipops do not, so the
    tiers stay in the expert order: agreement with the hierarchy reads as heights falling left to right, and
    each departure is an out-of-order height rather than something recovered from the rank labels. The
    main-text WoE figure draws its Tahoe panel the same way. The derived (response) order is still computed
    and returned for the order table in woe_structure.md.

    **The 1.0 line is an identity, not a measurement.** Dividing by the episode mean forces every episode
    to average exactly 1, so the count-weighted mean of the plotted tiers is 1.0000000000 for every model
    in every domain. Two things follow. It is a legitimate anchor -- above 1.0 means "this tier moves the
    model more than its average stub" -- but it is NOT evidence about the model. And normalization
    discards magnitude entirely: models are pinned to the same mean, so two models with identical shape
    and 10x different |WoE| plot identically here. That is the right trade for a panel about ORDERING, but
    it means heights must not be read across models as "moves more"; `relevant - control` and the raw
    deciban table carry magnitude.

    Returns the derived order per domain so it can go in the sidecar beside the Spearman table.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator

    # L_ is wide because a 40-45 deg label anchored at the FIRST tick extends left past the axes; at
    # L_=0.56 Tahoe's leading label ran off the canvas and was silently cut ("he drug, other cell").
    FIGW, R_, L_, GAP = TEXT_WIDTH_IN, 0.10, 0.72, 0.34
    PW = (FIGW - L_ - R_ - 2 * GAP) / 3
    # Laid out bottom-up so the rotated tick labels get a reserved band instead of overprinting the
    # caption, which is what happened when the caption was positioned by arithmetic on the margins.
    PH, TOP, XLAB, CAP, LEG = 1.80, 0.40, 1.02, 0.20, 0.46
    FIGH = TOP + PH + XLAB + CAP + LEG
    PANEL_Y = LEG + CAP + XLAB
    fig = plt.figure(figsize=(FIGW, FIGH))

    profiles_all = {k: {d: tier_profile(recs_by_model[k], d) for d in DOMAINS} for k in keys}
    # One shared y-range across panels, so the 1.0 anchor sits at the same height in all three and the
    # staircases are comparable by eye. Per-panel autoscaling put Tahoe's floor at 0.6 and GWAS's at 0.75,
    # which made equal drops look unequal.
    allv = [v for k in keys for d in DOMAINS for v in profiles_all[k][d].values()]
    ylo, yhi = min(min(allv), 1.0) * 0.85, max(allv) * 1.12  # log2 axis: multiplicative margins

    orders: dict[str, list[str]] = {}
    n = len(keys)
    slot = 0.82 / n  # a tier column's usable width, one fixed slot per model
    for i, d in enumerate(DOMAINS):
        prof_d = {k: profiles_all[k][d] for k in keys if profiles_all[k][d]}
        orders[d] = empirical_order(prof_d, d)  # the data's order, for the table; the plot uses HIER
        levels = [t for t in HIER[d] if any(t in p for p in prof_d.values())]
        ax = fig.add_axes(((L_ + i * (PW + GAP)) / FIGW, PANEL_Y / FIGH, PW / FIGW, PH / FIGH))
        ax.axhline(1.0, color=GRAY, ls="--", lw=1.0, zorder=1)
        x = np.arange(len(levels))
        for b in x[:-1] + 0.5:  # faint tier boundaries: the grouping unit is the tier
            ax.axvline(b, color=GREY["rule"], lw=0.5, zorder=0)
        for j, key in enumerate(keys):
            ys = np.array([prof_d.get(key, {}).get(t, np.nan) for t in levels], dtype=float)
            xo = x + (j - (n - 1) / 2) * slot
            col = plot_color(ROSTER[key])
            ax.vlines(xo, 1.0, ys, color=col, lw=0.9, zorder=3)
            ax.plot(xo, ys, ls="none", marker="o", ms=2.6, color=col, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"[{HIER[d].index(t) + 1}] {TIER_SHORT.get(t, t)}" for t in levels],
            fontsize=5.6,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_yscale("log", base=2)
        ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1.0, 2.0]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(lambda v, _: {0.5: "0.5\u00d7", 1.0: "1\u00d7", 2.0: "2\u00d7"}.get(v, f"{v:g}"))
        ax.tick_params(labelsize=7)
        ax.tick_params(axis="x", labelsize=5.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(dn.domain(d), fontsize=9.5)
        if i == 0:
            ax.set_ylabel("Normalized WoE", fontsize=8)
        else:
            ax.tick_params(labelleft=False)

    handles = [
        Line2D([], [], color=plot_color(ROSTER[k]), lw=4.0, solid_capstyle="butt", label=ROSTER[k].label) for k in keys
    ]
    handles.append(Line2D([], [], color=GRAY, ls="--", lw=1.0, label="episode mean (= 1 by construction)"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=int(np.ceil(len(handles) / 2)),
        frameon=False,
        fontsize=7.0,
        handlelength=1.6,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.010),
    )
    _save(fig, out, "woe_staircase", tight=False)
    return orders


def fig_directional_bars(series, keys, out: Path) -> None:
    """Mean SIGNED WoE by verdict — grouped bars with clustered-bootstrap CIs, all three domains.

    The paper's directional panel. Signed violins were tried first and dropped: they carry spread, which
    the bars do not, but at seven models x three verdicts the shapes are unreadable and the comparison the
    panel exists to make — does support push belief UP and refutation DOWN, and by how much — is buried.
    A bar's height is the quantity of interest and the whisker is its uncertainty, which is what a reader
    actually needs here. `--legacy-panels` still renders the violins for anyone who wants the spread.

    GWAS is included, drawn with only its two categories, because
    showing that the domain has no refuting slot states its structure where dropping the domain hid it.
    And Tahoe's `aggregate` is excluded: it is an evidence KIND rather than a strength level, so it does
    not belong on this axis (it stays in the tier hierarchy, where it is a tier).
    """
    import matplotlib.pyplot as plt

    FIGW, R_, L_, GAP = TEXT_WIDTH_IN, 0.10, 0.62, 0.36
    PW = (FIGW - L_ - R_ - 2 * GAP) / 3
    PH, TOP, BOT, CAP, LEG = 1.88, 0.38, 0.40, 0.16, 0.46
    FIGH = TOP + PH + BOT + CAP + LEG
    PANEL_Y = LEG + CAP + BOT
    fig = plt.figure(figsize=(FIGW, FIGH))

    n = len(keys)
    bw = 0.80 / n
    offs = np.linspace(-0.80 / 2 + bw / 2, 0.80 / 2 - bw / 2, n)

    def limit_for(doms) -> float:
        """Symmetric y-limit covering every bar and whisker in `doms`."""
        v = [
            series[k]["directional"].get(d, {}).get(c, {}).get(b, np.nan)
            for k in keys
            for d in doms
            for c in DIRECTIONAL_VERDS[d]
            for b in ("point", "lo", "hi")
        ]
        v = [x for x in v if not np.isnan(x)]
        return max(abs(min(v)), abs(max(v))) * 1.12 if v else 1.0

    # PER-DOMAIN y. GWAS's directional effects are ~0.8 db against Tahoe's 3.8 and DepMap's 4.6, so on a
    # shared axis its panel is a flat line and nothing within it can be read. Scaling per domain makes each
    # panel's own model comparison legible at the cost of cross-domain height comparison. The y-axis ticks stay
    # visible and labelled on EVERY panel rather than hidden on the 2nd and 3rd as a shared axis would allow --
    # that is what tells a reader the scales differ, now that the caption no longer says so. Symmetric about
    # zero in every panel, so the sign still reads by side.
    lims = {d: limit_for([d]) for d in DOMAINS}

    for i, d in enumerate(DOMAINS):
        cats = [v for v in DIRECTIONAL_VERDS[d] if any(v in series[k]["directional"].get(d, {}) for k in keys)]
        ax = fig.add_axes(((L_ + i * (PW + GAP)) / FIGW, PANEL_Y / FIGH, PW / FIGW, PH / FIGH))
        ax.axhline(0.0, color=GRAY, lw=0.9, zorder=1)  # belief unmoved — the sign flips across this line
        for j, key in enumerate(keys):
            got = series[key]["directional"].get(d, {})
            for xi, c in enumerate(cats):
                rec = got.get(c)
                if rec is None:
                    continue
                x = xi + offs[j]
                ax.bar(x, rec["point"], bw, color=plot_color(ROSTER[key]), linewidth=0, zorder=3)
                # Interval drawn at its own coordinates, matching render_grouped_metrics.draw_metric_panel:
                # a signed quantity makes +/- offsets easy to get backwards, and a one-sided whisker reads
                # as a drawing error rather than as an interval.
                if not (np.isnan(rec["lo"]) or np.isnan(rec["hi"])):
                    ax.plot([x, x], [rec["lo"], rec["hi"]], color=INK, lw=0.7, zorder=5)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_ylim(-lims[d], lims[d])
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([VERD_PRETTY.get(c, c) for c in cats], fontsize=6.6)
        # No bottom ticks: the bottom spine is hidden (zero is drawn as its own rule inside the axes), so
        # tick marks would float below the panel with no axis line to sit on.
        ax.tick_params(labelsize=7, bottom=False)
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.set_title(dn.domain(d), fontsize=9.5)
        if i == 0:
            ax.set_ylabel("mean signed WoE  (decibans)", fontsize=8)
        if d == "gwas":
            ax.text(
                0.98,
                0.03,
                "no refuting category:\nsignificance-only verdict",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.6,
                color=GRAY,
                style="italic",
            )

    handles = [plt.Rectangle((0, 0), 1, 1, color=plot_color(ROSTER[k]), alpha=0.95) for k in keys]
    fig.legend(
        handles,
        [ROSTER[k].label for k in keys],
        loc="lower center",
        ncol=int(np.ceil(len(keys) / 2)),
        frameon=False,
        fontsize=7.0,
        handlelength=1.4,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.010),
    )
    _save(fig, out, "woe_directional", tight=False)


def fig_directional_violin(recs_by_model, keys, out: Path) -> None:
    """SIGNED WoE by verdict — violins, one panel per domain, all three domains.

    Supersedes two legacy panels at once. `woe_by_verdict` showed a distribution but unsigned, so it could
    not say whether refuting evidence pushed belief DOWN; `woe_directional` showed the sign but only as a
    mean with a CI, which hides spread and bimodality. Signed violins carry both, so the pair collapses
    into one panel strictly more informative than either.
    """
    import matplotlib.pyplot as plt

    FIGW, R_, L_, GAP = TEXT_WIDTH_IN, 0.10, 0.62, 0.36
    PW = (FIGW - L_ - R_ - 2 * GAP) / 3
    PH, TOP, BOT, CAP, LEG = 2.00, 0.38, 0.46, 0.24, 0.48
    FIGH = TOP + PH + BOT + CAP + LEG
    PANEL_Y = LEG + CAP + BOT
    fig = plt.figure(figsize=(FIGW, FIGH))
    slot = 0.80
    offs = np.linspace(-slot / 2, slot / 2, len(keys))
    vw = slot / len(keys) * 0.92

    # ONE shared, CLIPPED y-range. Left to autoscale, each panel is set by its widest single series, and
    # that series is always Qwen base (direct): its p1/p99 is -28.8/+37.5 db against direct RL's
    # -3.6/+5.9, and it reaches the +/-60 db EPS saturation clamp. One row's spread then compresses the
    # other six into an indistinguishable band around zero -- which is the whole comparison.
    #
    # The limit is a quantile of the pooled |WoE| rather than a hand-picked number, so it tracks the data
    # if the roster changes. Clipping crops the drawing only; each violin's KDE is still estimated on its
    # full sample, so no shape near zero is altered. The cost is that a heavy-tailed series is visibly cut,
    # so how much is cut gets stated in the figure rather than left for the reader to infer.
    pooled = np.concatenate(
        [np.array([r["woe"] for r in recs_by_model[k]], dtype=float) for k in keys if recs_by_model[k]]
    )
    ylim = float(np.ceil(np.percentile(np.abs(pooled), CLIP_PCT) / 5.0) * 5.0)
    frac_out = {k: float(np.mean(np.abs([r["woe"] for r in recs_by_model[k]]) > ylim)) for k in keys}
    worst = max(frac_out, key=lambda k: frac_out[k])

    for i, d in enumerate(DOMAINS):
        cats = [
            v
            for v in DIRECTIONAL_VERDS[d]
            if any(any(r["verdict"] == v and r["domain"] == d for r in recs_by_model[k]) for k in keys)
        ]
        ax = fig.add_axes(((L_ + i * (PW + GAP)) / FIGW, PANEL_Y / FIGH, PW / FIGW, PH / FIGH))
        ax.axhline(0.0, color=GRAY, lw=0.9, zorder=1)  # belief unmoved — the sign flips across this line
        for j, key in enumerate(keys):
            for xi, c in enumerate(cats):
                vals = np.array(
                    [r["woe"] for r in recs_by_model[key] if r["domain"] == d and r["verdict"] == c], dtype=float
                )
                if len(vals) < 2:
                    continue
                parts = ax.violinplot([vals], positions=[xi + offs[j]], widths=vw, showextrema=False)
                for b in parts["bodies"]:
                    b.set_facecolor(plot_color(ROSTER[key]))
                    b.set_alpha(0.85)
                    b.set_edgecolor(INK)
                    b.set_linewidth(0.25)
                ax.hlines(float(vals.mean()), xi + offs[j] - vw / 2, xi + offs[j] + vw / 2, color=INK, lw=0.9, zorder=6)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([VERD_PRETTY.get(c, c) for c in cats], fontsize=6.4)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(dn.domain(d), fontsize=9.5)
        if i == 0:
            ax.set_ylabel("signed WoE per stub  (decibans)", fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        if d == "gwas":
            # Name the absent category IN the panel. A reader who does not know GWAS's verdict rule would
            # otherwise read two violins as an omission rather than as the domain's actual structure.
            ax.text(
                0.98,
                0.02,
                "no refuting category:\nsignificance-only verdict",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.6,
                color=GRAY,
                style="italic",
            )

    handles = [plt.Rectangle((0, 0), 1, 1, color=plot_color(ROSTER[k]), alpha=0.85) for k in keys]
    fig.legend(
        handles,
        [ROSTER[k].label for k in keys],
        loc="lower center",
        ncol=int(np.ceil(len(keys) / 2)),
        frameon=False,
        fontsize=7.0,
        handlelength=1.4,
        columnspacing=1.4,
        bbox_to_anchor=(0.5, 0.010),
    )
    # How much is cut, named per the worst-affected row. It is not
    # boilerplate that a caption can restate from the method -- it is measured from these data, and
    # without it the clip reads as the data's real extent. So it is REPORTED rather than dropped: the
    # caption has to carry these numbers, and they have to come from the render that produced the figure.
    print(
        f"[woe] directional_violin: y clipped at +/-{ylim:.0f} dB; "
        f"{100 * float(np.mean(np.abs(pooled) > ylim)):.1f}% of stubs outside overall, "
        f"{100 * frac_out[worst]:.0f}% for {ROSTER[worst].label} (worst). Put these in the caption."
    )
    _save(fig, out, "woe_directional_violin", tight=False)


def fig_by_verdict(recs_by_model, keys, out: Path) -> None:
    """|WoE| distribution by stub verdict — grouped VIOLINS, reproducing the source figure."""
    import matplotlib.pyplot as plt

    fig, flat = _panels(len(DOMAINS), 6.4, 4.9)
    slot = 0.78
    offs = np.linspace(-slot / 2, slot / 2, len(keys))
    vw = slot / len(keys) * 0.9
    for ax, d in zip(flat, DOMAINS):
        cats = [
            v
            for v in VERDS[d]
            if any(any(r["verdict"] == v and r["domain"] == d for r in recs_by_model[k]) for k in keys)
        ]
        for j, key in enumerate(keys):
            e = ROSTER[key]
            for i, c in enumerate(cats):
                vals = np.array([r["abswoe"] for r in recs_by_model[key] if r["domain"] == d and r["verdict"] == c])
                if len(vals) < 2:
                    continue
                parts = ax.violinplot([vals], positions=[i + offs[j]], widths=vw, showextrema=False)
                for b in parts["bodies"]:
                    b.set_facecolor(plot_color(e))
                    b.set_alpha(0.82)
                    b.set_edgecolor(INK)
                    b.set_linewidth(0.4)
                ax.hlines(float(vals.mean()), i + offs[j] - vw / 2, i + offs[j] + vw / 2, color=INK, lw=1.3, zorder=6)
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([VERD_PRETTY.get(c, c) for c in cats], fontsize=9)
        ax.set_title(f"{dn.domain(d)}", fontsize=11)
        ax.set_xlabel("stub verdict (statistical strength)", fontsize=9, color=GRAY)
        ax.set_ylabel("|WoE| per stub  (decibans)", fontsize=9, color=GRAY)
        _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=plot_color(ROSTER[k]), alpha=0.82) for k in keys]
    fig.legend(
        handles, [ROSTER[k].label for k in keys], loc="lower center", ncol=min(len(keys), 4), frameon=False, fontsize=9
    )
    fig.suptitle("Weight of evidence by stub verdict — multi-domain TEST set (sensitivity sweep)", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    _save(fig, out, "woe_by_verdict")


def fig_directional(series, keys, out: Path) -> None:
    """mean SIGNED WoE by verdict, grouped bars with clustered-bootstrap CI. GWAS excluded (no refuting)."""
    fig, flat = _panels(len(DIRECTIONAL_DOMAINS), 6.4, 4.9)
    n = len(keys)
    bw = 0.78 / n
    offs = np.linspace(-0.78 / 2 + bw / 2, 0.78 / 2 - bw / 2, n)
    for ax, d in zip(flat, DIRECTIONAL_DOMAINS):
        cats = VERDS[d]
        for j, key in enumerate(keys):
            e = ROSTER[key]
            got = series[key]["directional"].get(d, {})
            mus = [got.get(v, {}).get("point", np.nan) for v in cats]
            los = [max(0.0, (got.get(v, {}).get("point", np.nan) - got.get(v, {}).get("lo", np.nan))) for v in cats]
            his = [max(0.0, (got.get(v, {}).get("hi", np.nan) - got.get(v, {}).get("point", np.nan))) for v in cats]
            ax.bar(
                np.arange(len(cats)) + offs[j],
                mus,
                bw,
                color=plot_color(e),
                alpha=0.9,
                edgecolor=INK,
                lw=0.5,
                label=e.label,
                yerr=[los, his],
                capsize=2,
                error_kw={"ecolor": GREY["note"], "elinewidth": 1},
            )
        ax.axhline(0, color=GREY["rule"], lw=1)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([VERD_PRETTY.get(c, c) for c in cats], fontsize=9)
        ax.set_title(f"{dn.domain(d)}", fontsize=11)
        ax.set_xlabel("stub verdict", fontsize=9, color=GRAY)
        ax.set_ylabel("mean signed WoE  (decibans)", fontsize=9, color=GRAY)
        _style(ax)
    flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Directional correctness: signed WoE by verdict — multi-domain TEST set (sensitivity sweep)\n"
        "supported should push belief up, refuted down; GWAS omitted (significance-only, no refuting stubs)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save(fig, out, "woe_directional")


def fig_by_position(series, keys, out: Path) -> None:
    """mean |WoE| vs stub position, macro over domains — primacy / saturation. Its own figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for key in keys:
        s, e = series[key], ROSTER[key]
        pts = s["position"]
        xs = [x["position"] for x in pts]
        ax.plot(xs, [x["point"] for x in pts], color=plot_color(e), lw=2.2, marker="o", ms=4, label=e.label)
        lo = np.array([x["lo"] for x in pts], float)
        hi = np.array([x["hi"] for x in pts], float)
        if np.isfinite(lo).any():
            ax.fill_between(xs, lo, hi, color=plot_color(e), alpha=0.13, lw=0)
    ax.set_xlabel("stub position j   (order shown to the model)", fontsize=10, color=GRAY)
    ax.set_ylabel("mean |WoE| per stub  (decibans)", fontsize=10, color=GRAY)
    ax.legend(fontsize=8.5, frameon=False)
    _style(ax)
    fig.suptitle("Weight of evidence by stub position — macro over 3 domains (sensitivity sweep)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out, "woe_by_position")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/sensitivity/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--tag", default="ctx*", help="cells to read; must be the dense sweep (see docstring)")
    ap.add_argument("--keys", default=",".join(ORDER), help="comma list of roster keys")
    ap.add_argument("--n-boot", type=int, required=True, help="episode-clustered draws; 0 = points only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--legacy-panels",
        action="store_true",
        help="also render the four superseded panels (by_tier, by_verdict, directional bars, by_position). "
        "Diagnostics only — the paper uses woe_staircase + woe_directional_violin.",
    )
    ap.add_argument("--out-dir", default="outputs/woe_structure")
    args = ap.parse_args()

    if args.tag == "primary":
        raise SystemExit(
            "--tag primary cannot produce a weight of evidence: WoE is a difference between two forecasts "
            "on the SAME episode at k-1 and k, and primary scores each episode at one k only."
        )
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if unknown := [k for k in keys if k not in ROSTER]:
        raise SystemExit(f"unknown roster keys {unknown}; known: {list(ROSTER)}")

    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    print(f"[woe] reading stub tiers/verdicts from the published set @ {args.revision[:8]} (verifies nesting)…")
    meta = stub_meta(args.testset, revision=args.revision)

    nb = args.n_boot
    series, recs_by_model, table = {}, {}, []
    for key in keys:
        recs, diag = woe_records(slug, ROSTER[key].eval_key, args.tag, root, meta)
        if not recs:
            print(f"[woe] skip {key}: no computable increments", file=sys.stderr)
            continue
        # Loud, and ONLY on the reliable signal. A scored episode with no stub pool cannot happen when the
        # pools and the cells come from the same revision, so this is the skew alarm. Warning on short pools
        # too would fire on every correct render (two episodes legitimately hold 15 stubs, not 16), and an
        # alarm that is always on is an alarm nobody reads.
        if diag["n_episodes_without_stub_meta"]:
            print(
                f"[woe] !! {ROSTER[key].label}: {diag['n_episodes_without_stub_meta']}/"
                f"{diag['n_episodes_scored']} scored episodes have NO stub pool. The stub metadata and the "
                f"cells describe different episodes — check that --revision ({args.revision[:8]}) is the "
                f"revision these cells were scored against.",
                file=sys.stderr,
            )
        recs_by_model[key] = recs
        gap = (
            macro_ci(
                recs,
                gap_stat,
                cluster_key="episode_id",
                n_boot=nb,
                seed=args.seed,
                stat_id="woe:relevant_minus_control",
            )
            if nb
            else (macro_point(recs, gap_stat), float("nan"), float("nan"))
        )
        positions = []
        for j in sorted({r["position"] for r in recs}):
            sub = [r for r in recs if r["position"] == j]
            pt, lo, hi = (
                macro_ci(
                    sub,
                    _mean("abswoe"),
                    cluster_key="episode_id",
                    n_boot=max(nb // 4, 100),
                    seed=args.seed,
                    stat_id=f"woe:abswoe:pos{j}",
                )
                if nb
                else (macro_point(sub, _mean("abswoe")), float("nan"), float("nan"))
            )
            positions.append({"position": j, "point": pt, "lo": lo, "hi": hi, "n": len(sub)})
        tier_means, rho = {}, {}
        for d in DOMAINS:
            dd = [r for r in recs if r["domain"] == d]
            if not dd:
                continue
            tier_means[d] = {
                t: float(np.mean([r["abswoe"] for r in dd if r["tier"] == t]))
                for t in HIER[d]
                if any(r["tier"] == t for r in dd)
            }
            rank = {t: i for i, t in enumerate(HIER[d])}
            known = [r for r in dd if r["tier"] in rank]
            # Negated so POSITIVE means aligned: |WoE| falls as the relevance rank rises.
            rho[d] = -_spearman([rank[r["tier"]] for r in known], [r["abswoe"] for r in known])
        # Signed WoE per (domain, verdict): a within-domain clustered bootstrap, not macro — the verdict
        # vocabularies differ per domain, so there is no shared axis to macro-average over.
        directional = {}
        for d in DOMAINS:  # all three: GWAS is drawn too, with no refuting category to draw
            per = {}
            for v in DIRECTIONAL_VERDS[d]:
                sub = [r for r in recs if r["domain"] == d and r["verdict"] == v]
                if not sub:
                    continue
                pt, lo, hi = (
                    macro_ci(
                        sub,
                        _mean("woe"),
                        cluster_key="episode_id",
                        n_boot=max(nb // 4, 100),
                        seed=args.seed,
                        stat_id=f"woe:signed:{d}:{v}",
                    )
                    if nb
                    else (macro_point(sub, _mean("woe")), float("nan"), float("nan"))
                )
                per[v] = {"point": pt, "lo": lo, "hi": hi, "n": len(sub)}
            directional[d] = per
        series[key] = {
            "label": ROSTER[key].label,
            "eval_key": ROSTER[key].eval_key,
            "diagnostics": diag,
            "relevant_minus_control": {"point": gap[0], "lo": gap[1], "hi": gap[2]},
            "position": positions,
            "tier_means": tier_means,
            "rho": rho,
            "directional": directional,
        }
        table.append((ROSTER[key].label, gap, rho, diag, directional))
        print(
            f"[woe] {ROSTER[key].label:24s} inc={diag['n_increments']:5d} clamp={diag['clamp_rate']:5.1%} "
            f"gap={gap[0]:+6.2f} [{gap[1]:+.2f},{gap[2]:+.2f}] db"
        )

    if not series:
        raise SystemExit(f"no '{args.tag}' cells under {eval_paths.result_dir(slug, '', root=root).parent}")
    keys = [k for k in keys if k in series]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "woe_structure.json").write_text(
        json.dumps({"testset": args.testset, "tag": args.tag, "units": "decibans", "rows": series}, indent=2)
    )

    lines = [
        "| Model | relevant − control (db) | " + " | ".join(f"ρ {dn.domain(d)}" for d in DOMAINS) + " |",
        "|---|---|" + "|".join(["---"] * len(DOMAINS)) + "|",
    ]
    for lbl, gap, rho, _diag, _dir in table:
        rs = " | ".join(f"{rho.get(d, float('nan')):+.2f}" for d in DOMAINS)
        lines.append(f"| {lbl} | {gap[0]:+.2f} [{gap[1]:+.2f}, {gap[2]:+.2f}] | {rs} |")
    lines += [
        "",
        "`relevant − control` = mean |WoE| on target-relevant stubs minus mean |WoE| on control stubs, "
        "macro-averaged over domains; 95% CI from an episode-clustered bootstrap within domain. "
        "Positive = the model moves more on relevant evidence than on unrelated evidence.",
        "",
        "ρ = Spearman correlation between a stub's tier-relevance rank and |WoE|, signed so **positive "
        "means aligned** with the domain hierarchy. Tiers and their order are the committed per-domain "
        "hierarchies; depmap's `molecular_*` context types are collapsed into the source's single "
        "`molecular` tier.",
        "",
        "### Directional correctness — mean signed WoE (decibans)",
        "",
        "| Model | " + " | ".join(f"{dn.domain(d)} {v}" for d in DIRECTIONAL_DOMAINS for v in VERDS[d]) + " |",
        "|---|" + "|".join(["---"] * sum(len(VERDS[d]) for d in DIRECTIONAL_DOMAINS)) + "|",
    ]
    for lbl, _gap, _rho, _diag, dirn in table:
        cells = [
            f"{dirn.get(d, {}).get(v, {}).get('point', float('nan')):+.2f}"
            for d in DIRECTIONAL_DOMAINS
            for v in VERDS[d]
        ]
        lines.append(f"| {lbl} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "GWAS is omitted: its verdict is significance-only, so it has no refuting category and a signed "
        "figure would be meaningless (same reason `render_gwas_woe.py` omits it).",
        "",
        f"|WoE| saturates near {2 * abs(float(logit(EPS))) * DECIBAN:.0f} db (EPS={EPS}); clamp rates: "
        + ", ".join(f"{lbl} {diag['clamp_rate']:.1%}" for lbl, _g, _r, diag, _d in table)
        + ".",
    ]
    (out_dir / "woe_structure.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    # The two panels the paper uses. `fig_staircase` returns the DERIVED tier order, which is appended to
    # the sidecar next to the Spearman table -- the table stays scored against the CURATED hierarchy, so
    # the pair is exactly the comparison the section makes.
    orders = fig_staircase(recs_by_model, keys, out_dir)
    fig_directional_bars(series, keys, out_dir)
    _append_order_table(out_dir, orders)

    # Legacy panels, off by default. `woe_by_tier` is superseded by the staircase; `woe_by_verdict` and
    # `woe_directional` are jointly superseded by the signed violins (unsigned-with-spread and
    # signed-without-spread respectively); `woe_by_position` is deferred to the causal position-bias
    # analysis, which answers the same question directly. Kept because they are the diagnostics to reach
    # for when one of the new panels looks wrong.
    if args.legacy_panels:
        fig_by_tier(series, keys, out_dir)
        fig_by_verdict(recs_by_model, keys, out_dir)
        fig_directional_violin(recs_by_model, keys, out_dir)
        fig_by_position(series, keys, out_dir)
    print(f"[woe] data -> {out_dir / 'woe_structure.json'}, table -> {out_dir / 'woe_structure.md'}")


if __name__ == "__main__":
    main()
