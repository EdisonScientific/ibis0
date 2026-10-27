"""Renderer for depmap_meta episodes (gene-essentiality evidence deliberation).

Renders the structured stub dicts produced by scripts/data_pipeline/build_depmap_dataset.py into
model prompts, matching the tahoe_meta belief-model prompt format as closely as the domain allows so
that cross-domain comparisons measure generalization rather than prompt drift: the same emphatic system prompt
structure, CONTEXT-first / TARGET-HYPOTHESIS-last layout, per-stub relevance headers, and full-context
evidence cards with structured result blocks.
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

DEFAULT_DATASET_PATH = "hf://EdisonScientific/EDB-Bio/depmap"

# Only the domain preamble lives here; the shared reasoning/direct answer-format endings and the
# style->prompt selector come from ibis_env_utils.prompts. `SYSTEM_PROMPT` (reasoning) tells the
# model to think first — matching thinking-mode RL runs (`enable_thinking=true`) — while
# `SYSTEM_PROMPT_DIRECT` forbids any prose, matching direct runs (`enable_thinking=false`).
_SYSTEM_PROMPT_PREAMBLE = (
    "You are a scientific reasoning assistant analyzing cancer gene-dependency evidence from DepMap, a "
    "large-scale project of genome-wide CRISPR knockout screens of cancer cell lines with matched "
    "molecular and pharmacological profiles.\n"
    "How the evidence was generated:\n"
    "  - Each cell line is screened by pooled genome-wide CRISPR: every gene is knocked out (one gene per "
    "cell), the culture grows ~3 weeks, and cells whose knocked-out gene was required for proliferation "
    "drop out of the pool.\n"
    "  - The depletion of each gene is summarized by the Chronos gene-effect score; a calibrated "
    "P(dependency) is derived from it.\n"
    "  - Matched profiles, measured independently of the screen, give each line's baseline mRNA "
    "expression, copy number, somatic mutations, and (for some) drug sensitivity.\n"
    "Quantity definitions:\n"
    "  - Chronos gene-effect: a CRISPR-knockout fitness score, scaled so 0 = no effect on growth and "
    "-1 = as lethal as a typical core-essential gene (more negative = more essential; positive = knockout "
    "speeds growth). Analogous to a log2 fold-change.\n"
    "  - P(dependency): the calibrated probability in [0,1] that the gene is a true dependency in that "
    "cell line, inferred from the Chronos score. Analogous to statistical significance.\n"
    "You will be shown:\n"
    # Deliberately does NOT enumerate the context types (same-gene / pathway / molecular / drug /
    # control): part of the task is for the model to infer each stub's relevance rather than be
    # handed the taxonomy up front, as in gwas_meta. Mirrors
    # tahoe_meta, whose system prompt likewise describes CONTEXT generically; the per-stub relation
    # headers are kept (tahoe keeps them too). The measurement modalities are still described in the
    # "How the evidence was generated" section above.
    "  1. CONTEXT — prior measurements, each labeled by its relation to the target.\n"
    '  2. A TARGET HYPOTHESIS — a claim of the form "gene G is a dependency in cell line C".\n'
    "Weigh the evidence and report your degree of belief that a held-out genome-wide CRISPR screen would "
    "support the target hypothesis. Supported means the screen scores G at Chronos gene-effect <= -0.5 "
    "AND P(dependency) >= 0.5.\n"
)

SYSTEM_PROMPT, SYSTEM_PROMPT_DIRECT = build_probability_system_prompts(_SYSTEM_PROMPT_PREAMBLE)


def system_prompt_for(prompt_style: str, *, prior_note: bool = False) -> str:
    """DepMap system prompt for a `prompt_style` ("direct" / "reasoning").

    Thin domain binding of `ibis_env_utils.select_system_prompt`; shared by the RL env
    (`load_environment`) and the classification-head trainer so both resolve a style identically.
    `prior_note=True` inserts the shared PRIOR_NOTE (0.5-prior anchor) before the answer format.
    """
    reasoning, direct = build_probability_system_prompts(_SYSTEM_PROMPT_PREAMBLE, prior_note=prior_note)
    return select_system_prompt(prompt_style, reasoning=reasoning, direct=direct)


_RELEVANCE = {
    "same_gene_same_lineage": "target gene knocked out in a different cell line of the same lineage",
    "same_gene_other_lineage": "target gene knocked out in a cell line of a different lineage",
    "coessential_pathway": "a co-essential pathway partner of the target gene, knocked out in the target cell line",
    "molecular_expression": "baseline mRNA expression of the target gene in the target cell line",
    "molecular_cn": "copy-number state of the target gene in the target cell line",
    "molecular_mutation": "somatic mutation status of the target gene in the target cell line",
    "pharmacology": "a drug targeting the target gene, profiled in the target cell line",
    "control": "an unrelated gene knocked out in the target cell line",
}


def relevance_header(s):
    ct = s.get("context_type")
    if ct == "pharmacology" and s.get("via_pathway_gene"):
        return "a drug targeting a pathway partner of the target gene, profiled in the target cell line"
    return _RELEVANCE.get(ct, ct or "evidence")


def _chronos_call(ch):
    if ch is None:
        return "no data"
    if ch <= -0.5:
        return "substantial fitness cost"
    if ch <= -0.1:
        return "mild fitness reduction"
    if ch <= 0.25:
        return "no clear fitness effect"
    return "knockout speeds growth — suppressor"


def _pdep_call(p):
    if p is None:
        return "no data"
    if p >= 0.5:
        return "likely a true dependency"
    if p >= 0.1:
        return "unlikely a true dependency"
    return "almost certainly not a dependency"


def _gene_block(gm, indent="      "):
    if not gm:
        return ""
    extra = "; ".join(
        x
        for x in [
            f"family: {gm.get('family')}" if gm.get("family") else None,
            f"Ensembl {gm.get('ensembl_id')}" if gm.get("ensembl_id") else None,
        ]
        if x
    )
    nm = gm.get("name") or gm.get("symbol")
    return f"{indent}Gene: {gm.get('symbol')} — {nm}" + (f" ({extra})" if extra else "")


def _cell_block(cm, indent="      "):
    if not cm:
        return ""
    bits = " / ".join(x for x in [cm.get("disease"), cm.get("subtype")] if x)
    demo = "; ".join(
        x
        for x in [
            f"{cm.get('sex')}" if cm.get("sex") else None,
            cm.get("primary_metastasis") if cm.get("primary_metastasis") else None,
        ]
        if x
    )
    line = (
        f"{indent}Cell line: {cm.get('cell_name')} (DepMap {cm.get('model_id')}) — {cm.get('lineage')}"
        + (f" / {bits}" if bits else "")
        + (f"; {demo}" if demo else "")
    )
    drv = cm.get("driver_mutations")
    drv = [] if drv is None else list(drv)
    if len(drv):
        line += f"\n{indent}  Driver mutations: {', '.join(drv)}"
    return line


def _drug_block(dm, indent="      "):
    if not dm:
        return ""
    kv = "; ".join(
        x
        for x in [
            f"targets: {dm.get('targets')}" if dm.get("targets") else None,
            f"MOA: {dm.get('moa')}" if dm.get("moa") else None,
            f"ChEMBL {dm.get('chembl_id')}" if dm.get("chembl_id") else None,
        ]
        if x
    )
    parts = [f"{indent}Drug: {dm.get('drug')}"]
    if kv:
        parts.append(f"{indent}  {kv}")
    if dm.get("smiles"):
        parts.append(f"{indent}  SMILES: {dm.get('smiles')}")
    return "\n".join(parts)


def render_stub(s, hide_result: bool = False):
    """One CONTEXT card: a modality-specific setup line/blocks plus the measured ``Result:`` block.

    ``hide_result=True`` omits the ``Result:`` block (the measured ``y``), leaving only the experiment
    setup ``x`` — used to render a candidate stub whose outcome is to be predicted.
    """
    g, cell, y = s.get("gene"), (s.get("cell_metadata") or {}).get("cell_name", s.get("cell")), s.get("y") or {}
    gb, cb = _gene_block(s.get("gene_metadata")), _cell_block(s.get("cell_metadata"))
    mod = s["modality"]
    if mod == "crispr":
        head = f"CRISPR knockout of {g} in {cell}"
        ch, pdep = y.get("chronos"), y.get("p_dependency")
        setup, res = (
            [gb, cb],
            (
                "      Result:\n"
                f"        Chronos gene-effect  {ch:+.2f}  ({_chronos_call(ch)})\n"
                f"        P(dependency)        {pdep:.2f}   ({_pdep_call(pdep)})"
            ),
        )
    elif mod == "expression":
        head = f"Baseline mRNA expression of {g} in {cell}"
        z = y.get("lineage_z")
        sig = (
            " (significantly high)"
            if z is not None and z > 1.5
            else " (significantly low)"
            if z is not None and z < -1.5
            else ""
        )
        setup, res = [gb, cb], f"      Result: log2(TPM+1) = {y.get('log2_tpm')}; z = {z:+.1f} vs lineage{sig}"
    elif mod == "copy_number":
        head = f"Copy number of the {g} locus in {cell}"
        setup, res = [gb, cb], f"      Result: relative copy number = {y.get('relative_cn')} ({y.get('call')})"
    elif mod == "mutation":
        head = f"Somatic mutation status of {g} in {cell}"
        det = (
            (
                f"{'hotspot ' if y.get('hotspot') else ''}{'damaging ' if y.get('damaging') else ''}mutation "
                f"{y.get('protein_change') or ''}".strip()
            )
            if y.get("status") == "mutated"
            else "wild-type"
        )
        setup, res = [gb, cb], f"      Result: {det}"
    elif mod == "pharmacology":
        dn = (s.get("drug_metadata") or {}).get("drug")
        head = f"Pharmacological sensitivity of {cell} to {dn}"
        v = y.get("viability_log2fc")
        sens = (
            "sensitive" if y.get("sensitive") else "resistant" if (v is not None and v > -0.3) else "modest sensitivity"
        )
        setup = [_drug_block(s.get("drug_metadata")), cb]
        res = f"      Result: PRISM viability log2FC = {v:+.2f} ({sens}; more negative = more cell killing)"
    else:
        return f"{mod} stub"
    parts = [head, *setup] + ([] if hide_result else [res])
    return "\n".join(x for x in parts if x)


def render_evidence_stub(idx: int, s: dict, hide_result: bool = False) -> list[str]:
    """One CONTEXT block: the ``[idx]`` relevance header line plus the modality-specific card.

    ``hide_result=True`` keeps the setup but drops the measured ``Result:`` block.
    """
    return [f"\n[{idx}] {relevance_header(s)}", render_stub(s, hide_result=hide_result)]


def render_target(row: dict) -> str:
    """TARGET HYPOTHESIS block: the dependency claim + the target gene/cell context."""
    tgm, tcm = row.get("target_gene_meta"), row.get("target_cell_meta")
    cell = (tcm or {}).get("cell_name", row.get("target_cell"))
    parts = [
        f"Gene {row['target_gene']} is a dependency (essential for fitness) in cell line {cell}: "
        f"a held-out genome-wide CRISPR knockout screen of {cell} would score gene "
        f"{row['target_gene']} at Chronos gene-effect <= -0.5 AND P(dependency) >= 0.5.",
        _gene_block(tgm, indent="  "),
        _cell_block(tcm, indent="  "),
    ]
    return "\n".join(parts)


def render_user(row, *, prompt_style: str = "reasoning"):
    """User message: CONTEXT evidence (each tagged by relevance) first, then the TARGET HYPOTHESIS
    last with its gene/cell context --- via the shared ``render_evidence_prompt`` skeleton.

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
    stubs of the (fixed, pre-shuffled) pool. ``label`` is single-target (DepMap asks one hypothesis
    per episode).
    """
    return build_single_target_record(
        row,
        context_size,
        task="depmap_evidence_deliberation",
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
    """Load the DepMap gene-essentiality evidence-deliberation environment.

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
