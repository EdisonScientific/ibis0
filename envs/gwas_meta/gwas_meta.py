"""Renderer for gwas_meta episodes (genetic-association evidence deliberation).

Renders the structured stub dicts produced by scripts/data_pipeline/build_gwas_dataset.py into model
prompts, matching the tahoe_meta / depmap_meta belief-model prompt format as closely as the domain
allows so that cross-domain comparisons measure generalization rather than prompt drift: the same emphatic
system prompt structure, CONTEXT-first / TARGET-HYPOTHESIS-last layout, per-stub relevance headers,
and full-context evidence cards with structured result blocks.
"""

import datasets
import verifiers as vf
from ibis_env_utils import (
    UNPARSEABLE_REWARD,
    Split,
    build_probability_system_prompts,
    build_single_target_record,
    compute_reward,
    context_size_for,
    load_dataset_split,
    make_evidence_rubric,
    parse_prob,
    render_evidence_prompt,
    select_system_prompt,
    validate_context_size_range,
)

# parse_prob / compute_reward are re-exported for callers that resolve them off the domain module
# (eval sweeps, the classification trainer, tests).
__all__ = ["parse_prob", "compute_reward", "render_user", "build_record", "load_environment", "system_prompt_for"]

DEFAULT_DATASET_PATH = "hf://EdisonScientific/EDB-Bio/gwas"

# Only the domain preamble lives here; the shared reasoning/direct answer-format endings and the
# style->prompt selector come from ibis_env_utils.prompts. `SYSTEM_PROMPT` (reasoning) tells the
# model to think first — matching thinking-mode RL runs (`enable_thinking=true`) — while
# `SYSTEM_PROMPT_DIRECT` forbids any prose, matching direct runs (`enable_thinking=false`).
_SYSTEM_PROMPT_PREAMBLE = (
    "You are an expert geneticist and epidemiologist analyzing genetic-association evidence from the "
    "NHGRI-EBI GWAS Catalog, a curated collection of published genome-wide association studies (GWAS) "
    "of human traits and diseases.\n"
    "How the evidence was generated:\n"
    "  - A GWAS genotypes many individuals and tests each common variant (SNP) for association with a "
    "trait, reporting an effect size and a p-value per SNP.\n"
    "  - An association is called genome-wide significant at p < 5e-8, the threshold that corrects for "
    "~1 million independent tests across the genome.\n"
    "  - Each study reports, for the lead SNP, the risk allele whose effect is measured, its frequency, "
    "and the direction/size of effect (odds ratio for disease traits, beta for quantitative traits).\n"
    "Quantity definitions:\n"
    "  - p-value: statistical significance of the SNP-trait association; < 5e-8 is genome-wide significant.\n"
    "  - Risk allele: the allele whose effect is reported; different studies may report opposite alleles "
    "for the same SNP, so direction is always stated relative to the named allele.\n"
    "  - OR (odds ratio, binary/disease traits): OR > 1 means the risk allele increases disease odds, "
    "OR < 1 means it decreases them (protective).\n"
    "  - Beta (quantitative traits): the named allele raises the trait when the direction is 'increase', "
    "lowers it when 'decrease'.\n"
    "  - RAF: risk-allele frequency in the study population.\n"
    "  - Mapped gene: gene assigned to the SNP by genomic position (not a causal claim); EFO parent: the "
    "Experimental Factor Ontology category grouping related traits.\n"
    "You will be shown:\n"
    # Deliberately does NOT enumerate the context tiers (same-gene / same-SNP / pathway / control):
    # part of the task is for the model to infer each stub's relevance rather than be handed the
    # taxonomy up front. Mirrors tahoe_meta, whose system prompt likewise
    # describes CONTEXT generically; the per-stub relation headers are kept (tahoe keeps them too).
    "  1. CONTEXT — prior GWAS associations, each labeled by its relation to the target.\n"
    "  2. A TARGET HYPOTHESIS — a claim that a SNP influences a trait in a stated direction.\n"
    "Weigh the evidence and report your degree of belief that a held-out GWAS would support the target "
    "hypothesis. Supported means the held-out study reports p < 5e-8 for this SNP-trait pair with a "
    "consistent direction of effect.\n"
)

SYSTEM_PROMPT, SYSTEM_PROMPT_DIRECT = build_probability_system_prompts(_SYSTEM_PROMPT_PREAMBLE)


def system_prompt_for(prompt_style: str, *, prior_note: bool = False) -> str:
    """GWAS system prompt for a `prompt_style` ("direct" / "reasoning").

    Thin domain binding of `ibis_env_utils.select_system_prompt`; shared by the RL env
    (`load_environment`) and the classification-head trainer so both resolve a style identically.
    `prior_note=True` inserts the shared PRIOR_NOTE (0.5-prior anchor) before the answer format.
    """
    reasoning, direct = build_probability_system_prompts(_SYSTEM_PROMPT_PREAMBLE, prior_note=prior_note)
    return select_system_prompt(prompt_style, reasoning=reasoning, direct=direct)


_RELEVANCE = {
    "same_gene_same_trait": "a different SNP in the target gene, associated with the target trait",
    "same_gene_related_trait": "a different SNP in the target gene, associated with a trait in the same EFO parent",
    "same_snp_related_trait": "the target SNP, associated with a different trait in the same EFO parent",
    "same_snp_unrelated_trait": "the target SNP, associated with a trait in a different EFO parent",
    "pathway_gene_same_trait": "a SNP in a pathway-neighbor gene, associated with the target trait",
    "control": "an unrelated SNP and trait (different gene, different EFO parent)",
}


def relevance_header(s):
    ct = s.get("context_type")
    return _RELEVANCE.get(ct, ct or "evidence")


def _pval_call(p):
    if p is None:
        return "no data"
    if p < 5e-8:
        return "genome-wide significant"
    if p < 1e-5:
        return "suggestive, below the genome-wide bar"
    return "not significant"


def _fmt_int(n):
    # parquet promotes int columns with any null to float, so coerce back to a clean integer string
    return f"{int(n):,}" if isinstance(n, (int, float)) and n == n else None


def _snp_block(s, indent="      "):
    sm = s.get("snp_metadata") or {}
    loc = ""
    if sm.get("chr") is not None and sm.get("pos") is not None:
        loc = f" — chr{sm.get('chr')}:{sm.get('pos')}"
    lines = [f"{indent}SNP: {s.get('snp')}{loc}"]
    d = sm.get("distance_to_target")
    if d is not None:
        lines.append(f"{indent}  Distance to target: {_fmt_int(d)} bp")
    if sm.get("risk_allele"):
        lines.append(f"{indent}  Risk allele: {sm.get('risk_allele')}")
    if sm.get("raf") is not None:
        lines.append(f"{indent}  RAF: {sm.get('raf')}")
    if s.get("gene"):
        lines.append(f"{indent}  Mapped gene: {s.get('gene')}")
    if sm.get("functional_class"):
        lines.append(f"{indent}  Functional class: {sm.get('functional_class')}")
    if sm.get("intergenic"):
        lines.append(f"{indent}  Intergenic: yes")
    return "\n".join(lines)


def _gene_context_block(gm, indent="      "):
    if gm is None:
        return ""
    lines = []
    # reactome_pathways round-trips through parquet as a numpy array, so avoid bare truthiness (an
    # empty array raises "ambiguous"); normalize to a list first.
    pw = gm.get("reactome_pathways")
    pw = [] if pw is None else list(pw)
    if len(pw):
        lines.append(f"{indent}  Reactome pathways: {', '.join(str(p) for p in pw)}")
    tissues = gm.get("gtex_top_tissues")
    if isinstance(tissues, str) and tissues:  # parquet nulls come back as NaN/None, not a string
        lines.append(f"{indent}  GTEx expression: {tissues}")
    if not lines:
        return ""
    return f"{indent}Gene context:\n" + "\n".join(lines)


def _study_block(sm, indent="      "):
    if not sm or not sm.get("accession"):
        return ""
    lines = [f"{indent}Study: {sm.get('accession')}"]
    if sm.get("ancestry"):
        lines.append(f"{indent}  Ancestry: {sm.get('ancestry')}")
    if sm.get("sample_size") is not None:
        lines.append(f"{indent}  Sample size: {_fmt_int(sm.get('sample_size'))}")
    if sm.get("platform"):
        lines.append(f"{indent}  Platform: {sm.get('platform')}")
    return "\n".join(lines)


def _result_block(y, indent="      "):
    if not y:
        return ""
    p = y.get("p_value")
    pstr = f"{p:.1e}" if p is not None else "N/A"
    ob = y.get("or_beta")
    obstr = f"{ob}" if ob is not None else "N/A"
    lines = [f"{indent}Result: p = {pstr} ({_pval_call(p)}); OR/beta = {obstr}"]
    if y.get("ci_text"):
        lines.append(f"{indent}  95% CI: {y.get('ci_text')}")
    if y.get("direction_phrase"):
        lines.append(f"{indent}  Direction: {y.get('direction_phrase')}")
    return "\n".join(lines)


def render_stub(s, hide_result: bool = False):
    """One CONTEXT evidence card: SNP identity, gene context, study, and the association result.

    ``hide_result=True`` omits the measured association result (the ``y`` block), leaving only the
    experiment setup ``x`` — used to render a candidate stub whose outcome is to be predicted.
    """
    head = f"{s.get('snp')} for {s.get('trait')}"
    parts = [
        head,
        _snp_block(s),
        _gene_context_block(s.get("gene_metadata")),
        _study_block(s.get("study_metadata")),
    ]
    if not hide_result:
        parts.append(_result_block(s.get("y")))
    return "\n".join(x for x in parts if x)


def _hypothesis_sentence(row):
    """Plain-language target hypothesis, matching the paper's example episode."""
    snp = row.get("target_snp")
    trait = row.get("target_trait")
    allele = (row.get("target_snp_meta") or {}).get("risk_allele") or row.get("target_allele")
    direction = row.get("target_direction")
    allele_clause = f" ({allele} allele)" if allele else ""
    if direction is not None and direction != 0:
        verb = "increases" if direction > 0 else "decreases"
        return f"SNP {snp}{allele_clause} {verb} {trait}."
    return f"SNP {snp}{allele_clause} is associated with {trait}."


def render_evidence_stub(idx: int, s: dict, hide_result: bool = False) -> list[str]:
    """One CONTEXT block: the ``[idx]`` relevance header line plus the SNP/gene/study/result card.

    ``hide_result=True`` keeps the setup (relevance/SNP/gene/study) but drops the measured result.
    """
    return [f"\n[{idx}] {relevance_header(s)}", render_stub(s, hide_result=hide_result)]


def render_target(row: dict) -> str:
    """TARGET HYPOTHESIS block: the plain-language hypothesis + the target SNP/gene/trait context."""
    tsm = row.get("target_snp_meta") or {}
    tgm = row.get("target_gene_meta")
    ttm = row.get("target_trait_meta") or {}
    parts = [
        _hypothesis_sentence(row),
        "A held-out GWAS would report p < 5e-8 for this SNP-trait pair with a consistent direction of effect.",
        _snp_block({"snp": row.get("target_snp"), "gene": row.get("target_gene"), "snp_metadata": tsm}, indent="  "),
    ]
    gc = _gene_context_block(tgm, indent="  ")
    if gc:
        parts.append(gc)
    trait_line = f"  Trait: {row.get('target_trait')}"
    if ttm.get("efo_parent"):
        trait_line += f"\n    EFO parent: {ttm.get('efo_parent')}"
    parts.append(trait_line)
    return "\n".join(parts)


def render_user(row, *, prompt_style: str = "reasoning"):
    """User message: CONTEXT evidence (each tagged by relevance) first, then the TARGET HYPOTHESIS
    last with its SNP/gene/trait context --- via the shared ``render_evidence_prompt`` skeleton.

    ``prompt_style`` is accepted for a uniform contract with the other domains but does not alter the
    user text (the output-format instruction lives in the system prompt only)."""
    return render_evidence_prompt(
        row["context_pool"],
        stub_renderer=render_evidence_stub,
        target_block=render_target(row),
    )


def build_record(row: dict, context_size: int, *, prompt_style: str = "reasoning") -> dict:
    """Pure record builder: a dataset row -> a verifiers example (see ``build_single_target_record``).

    Renders the prompt at the requested ``context_size`` by taking the first ``context_size``
    stubs of the (fixed, pre-shuffled) pool. ``label`` is single-target (one hypothesis per episode).
    """
    return build_single_target_record(
        row,
        context_size,
        task="gwas_evidence_deliberation",
        render_user=render_user,
        prompt_style=prompt_style,
        extra_info=lambda r: {"ood_group": r.get("ood_group")},
    )


def load_environment(
    data_path: str = DEFAULT_DATASET_PATH,
    context_size: int = 16,
    context_size_range: tuple[int, int] | None = None,
    context_seed: int = 0,
    reward_type: str = "likelihood",
    prompt_style: str = "direct",
    system_prompt: str | None = None,
    unparseable_reward: float = UNPARSEABLE_REWARD,
    log_lik_clip: float = -10.0,
) -> "vf.Environment":
    """Load the GWAS genetic-association evidence-deliberation environment.

    Reads a collection's per-split parquet files via ``ibis_env_utils.load_dataset_split``
    (``hf://<org>/<repo>/<config>`` or a local dir of ``train``/``val``/``test`` parquet
    files). ``train`` is loaded for training and ``val`` for validation/model selection;
    ``test`` is never loaded here — it is held out for final reporting.

    Context length: each episode renders the first K stubs of its pool. By default K is the fixed
    ``context_size``; set ``context_size_range=(lo, hi)`` to instead draw K per episode uniformly in
    ``{lo..hi}``, deterministically keyed by ``episode_id`` (+ ``context_seed``) so an episode's K is
    stable across runs/splits (see ``ibis_env_utils.context_size_for``).

    ``prompt_style`` selects the system prompt: ``"direct"`` (answer-only, for
    ``enable_thinking=false`` runs) or ``"reasoning"`` (think-first, for thinking-mode runs).
    An explicit ``system_prompt`` overrides the style. Defaults to ``"direct"``.
    """
    if system_prompt is None:
        system_prompt = system_prompt_for(prompt_style)
    context_size_range = validate_context_size_range(context_size_range)

    def render_records(split: Split) -> list[dict]:
        # The dataset consumes the uniform chat-`prompt` record shape (identical across all three
        # evidence-deliberation envs); `build_record`'s own `question`/`task` shape is kept for the
        # classification trainer and eval sweeps that read it directly.
        df = load_dataset_split(data_path, split)
        records = [
            build_record(row, context_size_for(row.get("episode_id"), context_size, context_size_range, context_seed))
            for row in df.to_dict(orient="records")
        ]
        return [
            {
                "prompt": [{"role": "user", "content": r["question"]}],
                "answer": r["answer"],
                "info": r["info"],
                "task": r["task"],
            }
            for r in records
        ]

    train_records = render_records(Split.TRAIN)
    val_records = render_records(Split.VAL)
    if not train_records:
        raise ValueError(f"no rows in the 'train' split of {data_path}")
    if not val_records:
        raise ValueError(f"no rows in the 'val' split of {data_path}")

    parser = vf.Parser()
    rubric = make_evidence_rubric(
        reward_type=reward_type,
        unparseable_reward=unparseable_reward,
        log_lik_clip=log_lik_clip,
        parser=parser,
    )

    return vf.SingleTurnEnv(
        dataset=datasets.Dataset.from_list(train_records),
        # verifiers' eval slot; we feed it the `val` (validation) split for model selection.
        eval_dataset=datasets.Dataset.from_list(val_records),
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
