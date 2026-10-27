"""Tahoe episode representation, rendering, and per-split loading.

The data layer shared by every Tahoe consumer. It holds the schema dataclasses (target
hypothesis, evidence/aggregate context, episode), the DESeq2 `compute_verdict` rule, the system
prompts + `PromptRenderer`, episode reconstruction from a dataset row (`_row_to_episode`), and
`load_rendered_split` — the single load-and-render entry point
(`load_dataset_split` -> `_row_to_episode` -> `PromptRenderer`) that the RL env (`tahoe_meta.py`),
the classification-head trainer, and the offline eval/analysis scripts all build on.

Kept separate from `tahoe_meta.py` so that module stays focused on the verifiers environment and
its reward/scoring functions, while this one owns how a Tahoe episode is represented and rendered.
"""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from ibis_env_utils import (
    PRIOR_NOTE,
    Split,
    build_single_target_record,
    context_size_for,
    load_dataset_split,
    render_evidence_prompt,
    validate_context_size_range,
)

# The two system prompts differ only in their final answer-format instructions (reasoning trace vs
# answer-only). Share the common preamble so the task framing — evidence provenance, column
# definitions, the support criterion — stays defined in one place. `SYSTEM_PROMPT` /
# `SYSTEM_PROMPT_DIRECT` below are pinned byte-for-byte (a Tahoe run is tied to its exact prompt); the
# env tests assert the composed strings.
_SYSTEM_PROMPT_PREAMBLE = """You are a scientific reasoning assistant analyzing drug-perturbation evidence from Tahoe-100M, a large-scale pseudobulk RNA-seq dataset measuring drug × cell-line × gene differential expression.

How the evidence was generated:
  - Cells from a given cell line are treated with a drug (or DMSO vehicle control).
  - Each cell's transcripts are captured by single-cell RNA-seq; counts are UMI-deduplicated per gene.
  - Single cells from the same (drug, cell_line, plate, replicate) bucket are *pseudobulked* by summing per-gene UMI counts.
  - For each gene, DESeq2 fits a negative-binomial GLM comparing treated vs control pseudobulk samples.

Column definitions (DESeq2 conventions):
  - log2FoldChange: shrunken log2(treated / control) of normalized counts. Positive = upregulated; negative = downregulated.
  - lfcSE:         standard error of the log2FoldChange estimate (uncertainty).
  - pvalue:        Wald test p-value for log2FoldChange ≠ 0.
  - padj:          Benjamini-Hochberg adjusted p-value (FDR-controlled across all genes in that drug × cell_line comparison).
  - baseMean:      mean of normalized counts across both treated AND control samples combined (an expression-level indicator, NOT a control baseline).
  - n_cells_trt / n_cells_ctrl: number of single cells pooled into the treated / control pseudobulk.

You will be shown:
  1. CONTEXT — prior experimental results: drug × cell-line × gene differential-expression measurements, plus aggregate perturbation summaries.
  2. A TARGET HYPOTHESIS — a claim of the form "drug D upregulates/downregulates gene G in cell line C".

Weigh the evidence and report your degree of belief that a held-out experiment would *support* the target hypothesis. Supported means padj < 0.05 AND |log2FoldChange| > 0.5 in the claimed direction."""


SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_PREAMBLE
    + """

Reason about what the context implies — for example: the drug's transcriptional footprint in this and nearby cell lines, whether mechanistically similar drugs show consistent effects on the target gene, whether the gene appears broadly responsive or quiet, the drug's aggregate perturbation strength in the target cell line, and any contradictions between evidence pieces. Then commit to a single probability in the closed interval [0, 1].

Think through the evidence carefully, then commit to a single probability in the closed interval [0, 1].

After your reasoning, output your probability in this exact format:
<probability>0.XX</probability>"""
)


SYSTEM_PROMPT_DIRECT = (
    _SYSTEM_PROMPT_PREAMBLE
    + """

Respond with ONLY a single line in this exact format — no reasoning, no explanation, no other text before or after:
<probability>0.XX</probability>"""
)


# --- Schema -----------------------------------------------------------------


@dataclass
class TargetHypothesis:
    """A target hypothesis to evaluate."""

    target_id: str
    drug: str
    cell_line: str
    gene: str
    direction: Literal["upregulates", "downregulates"]

    def to_string(self) -> str:
        return f"Drug {self.drug} {self.direction} gene {self.gene} in cell line {self.cell_line}"


@dataclass
class EvidenceTuple:
    """A single piece of evidence about a drug-cell-gene relationship."""

    tuple_id: str
    context_type: Literal[
        "same_drug_diff_cell",
        "diff_drug_same_cell",
        "same_moa_evidence",
        "negative_control",
    ]
    theta: str
    x: str
    y: dict
    c: Literal["supported", "refuted", "unresolved"]
    drug_metadata: dict
    cell_metadata: dict
    gene_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        expected_keys = {
            "gene",
            "log2FoldChange",
            "padj",
            "pvalue",
            "n_cells_trt",
            "n_cells_ctrl",
            "concentration",
        }
        for key in expected_keys:
            if key not in self.y:
                self.y[key] = None


@dataclass
class AggregateContext:
    """Aggregate perturbation statistics for a drug-cell pair (no per-gene data)."""

    tuple_id: str
    drug: str
    cell_line: str
    n_sig_genes_lfc05: int
    n_sig_genes_lfc10: int
    mean_abs_lfc: float
    median_abs_lfc: float
    perturbation_strength_percentile: float
    context_type: Literal["same_drug_cell_aggregate"] = "same_drug_cell_aggregate"
    drug_metadata: dict = field(default_factory=dict)
    cell_metadata: dict = field(default_factory=dict)


@dataclass
class TahoeEvidenceEpisode:
    """A complete episode for the Tahoe evidence deliberation task (single-target)."""

    episode_id: str
    split: Split
    target: TargetHypothesis
    context: list[EvidenceTuple | AggregateContext]
    label: int
    metadata: dict = field(default_factory=dict)


def compute_verdict(
    log2fc: float | None,
    padj: float | None,
    claimed_direction: str,
    lfc_threshold: float = 0.5,
) -> Literal["supported", "refuted", "unresolved"]:
    """Verdict for a (drug, cell, gene, direction) hypothesis given DE results.

    A hypothesis is `supported` iff padj < 0.05 AND |log2FC| > threshold AND the
    observed direction matches the claim. `refuted` means significant in the
    opposite direction. Anything else is `unresolved`.
    """
    if log2fc is None or padj is None:
        return "unresolved"
    if padj >= 0.05:
        return "unresolved"
    if abs(log2fc) <= lfc_threshold:
        return "unresolved"

    actual_direction = "upregulates" if log2fc > 0 else "downregulates"
    return "supported" if actual_direction == claimed_direction else "refuted"


# --- Prompt rendering -------------------------------------------------------


_CONTEXT_TYPE_EXPLAIN = {
    "same_drug_diff_cell": "same drug as target, different cell line, target gene",
    "diff_drug_same_cell": "different drug, same cell line as target, target gene",
    "same_moa_evidence": "drug with same mechanism of action as the target drug",
    "negative_control": "unrelated drug × cell line, included as a baseline",
    "same_drug_cell_aggregate": "genome-wide summary for the target drug × cell line",
}


def _has_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    return True


def _fmt(v) -> str:
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
            return f"{v:.3g}"
        return f"{v:.4g}"
    return str(v)


def _render_kv_block(label: str, items: list[tuple[str, object]], indent: str = "      ") -> list[str]:
    rows = [(k, v) for k, v in items if _has_value(v)]
    if not rows:
        return []
    lines = [f"    {label}"]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        lines.append(f"{indent}{k.ljust(width)}  {_fmt(v)}")
    return lines


def _drug_block(meta: dict) -> list[str]:
    if not meta:
        return []
    return _render_kv_block(
        f"Drug: {meta.get('drug', '?')}",
        [
            ("MOA (broad)", meta.get("moa-broad")),
            ("MOA (fine)", meta.get("moa-fine")),
            ("Targets", meta.get("targets")),
            ("Human-approved", meta.get("human-approved")),
            ("In clinical trials", meta.get("clinical-trials")),
            ("Notes", meta.get("gpt-notes-approval")),
            ("Canonical SMILES", meta.get("canonical_smiles")),
            ("PubChem CID", meta.get("pubchem_cid")),
        ],
    )


def _cell_block(meta: dict) -> list[str]:
    if not meta:
        return []
    return _render_kv_block(
        f"Cell line: {meta.get('cell_name', '?')}",
        [
            ("DepMap ID", meta.get("Cell_ID_DepMap")),
            ("Cellosaurus ID", meta.get("Cell_ID_Cellosaur")),
            ("Organ", meta.get("Organ")),
            ("Driver gene", meta.get("Driver_Gene_Symbol")),
            ("Driver variant type", meta.get("Driver_VarType")),
            ("Driver zygosity", meta.get("Driver_VarZyg")),
            ("Driver protein/cDNA effect", meta.get("Driver_ProtEffect_or_CdnaEffect")),
            ("Driver mechanism (LoF/GoF)", meta.get("Driver_Mech_InferDM")),
            ("Driver gene type", meta.get("Driver_GeneType_DM")),
        ],
    )


def _gene_block(meta: dict, gene_name: str | None = None) -> list[str]:
    name = (meta or {}).get("gene_symbol") or gene_name
    if not name:
        return []
    return _render_kv_block(
        f"Gene: {name}",
        [("Ensembl ID", (meta or {}).get("ensembl_id"))],
    )


def _experiment_block(y: dict) -> list[str]:
    conc = y.get("concentration")
    unit = y.get("concentration_unit")
    if _has_value(conc) and _has_value(unit):
        dose = f"{_fmt(conc)} {unit}"
    elif _has_value(conc):
        dose = _fmt(conc)
    else:
        dose = None
    return _render_kv_block(
        "Experiment:",
        [
            ("Dose", dose),
            ("Plate", y.get("plate")),
            ("n_cells treated", y.get("n_cells_trt")),
            ("n_cells control", y.get("n_cells_ctrl")),
            ("baseMean (avg normalized counts across both arms)", y.get("baseMean")),
        ],
    )


def _result_block(y: dict) -> list[str]:
    lfc = y.get("log2FoldChange")
    lfc_se = y.get("lfcSE")
    direction = None
    lfc_str = None
    if _has_value(lfc):
        direction = "upregulates" if lfc > 0 else "downregulates"
        lfc_str = f"{lfc:+.3f}"
        if _has_value(lfc_se):
            lfc_str += f" ± {lfc_se:.3f}"
        if direction:
            lfc_str += f"  ({direction})"
    rows = [
        ("log2FoldChange", lfc_str),
        ("Wald statistic", y.get("stat")),
        ("p-value (raw)", y.get("pvalue")),
        ("p-value (adjusted)", y.get("padj")),
    ]
    return _render_kv_block("Result:", rows)


TAHOE_EMPTY_CONTEXT_NOTICE = "(no prior evidence provided — base your judgement on prior knowledge only.)"


def render_evidence_stub(idx: int, stub: dict, hide_result: bool = False) -> list[str]:
    """One CONTEXT block for a Tahoe evidence/aggregate stub dict (the shared per-stub renderer).

    Dispatches on ``stub["kind"]``: an ``"aggregate"`` stub renders the genome-wide perturbation
    summary; anything else renders the per-gene DE evidence card (drug / cell / gene / experiment /
    result). The ``[idx]`` header carries the context-type explanation.

    ``hide_result=True`` keeps the drug/cell/gene setup but omits the measured outcome block (the
    genome-wide summary for aggregate stubs, or the experiment + DE result for evidence stubs) — used
    to render a candidate stub whose outcome is to be predicted.
    """
    ctype = stub.get("context_type")
    explain = _CONTEXT_TYPE_EXPLAIN.get(ctype, "")
    header = f"\n[{idx}] {explain}" if explain else f"\n[{idx}]"
    drug_meta = stub.get("drug_metadata") or {}
    cell_meta = stub.get("cell_metadata") or {}
    y = stub.get("y") or {}
    lines = [header]
    lines.extend(_drug_block(drug_meta))
    lines.extend(_cell_block(cell_meta))
    if stub.get("kind") == "aggregate":
        if not hide_result:
            lines.extend(
                _render_kv_block(
                    "Genome-wide perturbation summary:",
                    [
                        ("significant genes (|log2FC| > 0.5)", y.get("n_sig_genes_lfc05")),
                        ("significant genes (|log2FC| > 1.0)", y.get("n_sig_genes_lfc10")),
                        ("mean |log2FC|", y.get("mean_abs_lfc")),
                        ("median |log2FC|", y.get("median_abs_lfc")),
                        ("perturbation strength percentile", y.get("perturbation_strength_percentile")),
                    ],
                )
            )
    else:
        lines.extend(_gene_block(stub.get("gene_metadata") or {}, gene_name=y.get("gene")))
        if not hide_result:
            lines.extend(_experiment_block(y))
            lines.extend(_result_block(y))
    return lines


def render_target(row: dict) -> str:
    """TARGET HYPOTHESIS block: the drug×cell×gene claim + the target drug/cell/gene context."""
    parts = [
        f"Drug {row['target_drug']} {row['target_direction']} gene {row['target_gene']} "
        f"in cell line {row['target_cell']}.",
        "",
        *_drug_block(row.get("target_drug_meta") or {}),
        *_cell_block(row.get("target_cell_meta") or {}),
        *_gene_block(row.get("target_gene_meta") or {}, gene_name=row.get("target_gene")),
    ]
    return "\n".join(parts)


def render_user(row: dict, *, prompt_style: str = "reasoning") -> str:
    """User message via the shared ``render_evidence_prompt`` skeleton (context first, target last).

    ``row`` is a raw dataset row: ``context_pool`` is a list of stub dicts, and the target fields are
    ``target_drug`` / ``target_cell`` / ``target_gene`` / ``target_direction`` (+ ``target_*_meta``).
    ``prompt_style`` is accepted for a uniform contract but does not alter the user text (the
    output-format instruction lives in the system prompt only)."""
    return render_evidence_prompt(
        row.get("context_pool"),
        stub_renderer=render_evidence_stub,
        target_block=render_target(row),
        empty_context_notice=TAHOE_EMPTY_CONTEXT_NOTICE,
    )


def build_record(row: dict, context_size: int, *, prompt_style: str = "reasoning") -> dict:
    """A dataset row -> a single-target verifiers example (see ``build_single_target_record``)."""
    return build_single_target_record(
        row,
        context_size,
        task="tahoe_evidence_deliberation",
        render_user=render_user,
        prompt_style=prompt_style,
    )


def _context_to_stub(ctx: "EvidenceTuple | AggregateContext") -> dict:
    """Episode-dataclass context item -> the stub dict shape ``render_evidence_stub`` expects."""
    if isinstance(ctx, AggregateContext):
        return {
            "kind": "aggregate",
            "context_type": ctx.context_type,
            "drug_metadata": ctx.drug_metadata,
            "cell_metadata": ctx.cell_metadata,
            "y": {
                "n_sig_genes_lfc05": ctx.n_sig_genes_lfc05,
                "n_sig_genes_lfc10": ctx.n_sig_genes_lfc10,
                "mean_abs_lfc": ctx.mean_abs_lfc,
                "median_abs_lfc": ctx.median_abs_lfc,
                "perturbation_strength_percentile": ctx.perturbation_strength_percentile,
            },
        }
    return {
        "kind": "evidence",
        "context_type": ctx.context_type,
        "drug_metadata": ctx.drug_metadata,
        "cell_metadata": ctx.cell_metadata,
        "gene_metadata": ctx.gene_metadata,
        "y": ctx.y,
    }


def _episode_to_row(episode: TahoeEvidenceEpisode) -> dict:
    """A `TahoeEvidenceEpisode` -> the raw-row shape `render_user` renders (stub dicts + target)."""
    t = episode.target
    meta = episode.metadata or {}
    return {
        "context_pool": [_context_to_stub(c) for c in episode.context],
        "target_drug": t.drug,
        "target_cell": t.cell_line,
        "target_gene": t.gene,
        "target_direction": t.direction,
        "target_drug_meta": meta.get("target_drug_meta"),
        "target_cell_meta": meta.get("target_cell_meta"),
        "target_gene_meta": meta.get("target_gene_meta"),
    }


class PromptRenderer:
    """Renders episodes into prompts for the model.

    A thin adapter over the shared ``render_evidence_prompt`` pipeline (the same one the RL env and
    classification trainer use via ``render_user``): it converts an episode's typed context back into
    stub dicts and delegates, so analysis output stays byte-consistent with the env. ``mode="debug"``
    appends the hidden ground-truth label for eyeballing episodes.
    """

    def __init__(
        self,
        mode: Literal["compact", "debug"] = "compact",
        lfc_threshold: float = 0.5,
        prompt_style: Literal["reasoning", "direct"] = "reasoning",
    ):
        if prompt_style not in ("reasoning", "direct"):
            raise ValueError(f"unknown prompt_style: {prompt_style!r}")
        self.mode = mode
        self.lfc_threshold = lfc_threshold
        self.prompt_style = prompt_style

    def render(self, episode: TahoeEvidenceEpisode) -> str:
        if self.mode == "compact":
            return self._render_compact(episode)
        if self.mode == "debug":
            return self._render_debug(episode)
        raise ValueError(f"unknown render mode: {self.mode!r}")

    def _render_compact(self, episode: TahoeEvidenceEpisode) -> str:
        return render_user(_episode_to_row(episode), prompt_style=self.prompt_style)

    def _render_debug(self, episode: TahoeEvidenceEpisode) -> str:
        lines = [
            "=" * 80,
            f"EPISODE {episode.episode_id}  ({episode.split}, context_size={len(episode.context)})",
            "=" * 80,
            "",
            self._render_compact(episode),
            "",
            "-" * 40,
            "GROUND TRUTH (hidden from model)",
            "-" * 40,
            f"  {episode.target.target_id}: {episode.label}  "
            f"({'supported' if episode.label == 1 else 'not supported'})",
        ]
        return "\n".join(lines)


# --- Episode reconstruction + per-split loading -----------------------------


def _stub_to_context(stub: dict) -> EvidenceTuple | AggregateContext:
    drug_meta = dict(stub.get("drug_metadata") or {})
    cell_meta = dict(stub.get("cell_metadata") or {})
    gene_meta = dict(stub.get("gene_metadata") or {})
    y = dict(stub.get("y") or {})

    if stub.get("kind") == "aggregate":
        # drug/cell_line come from metadata that can legitimately be empty, so keep the soft `.get`.
        # The aggregate `y` stats are always written by the dataset builder; a missing one means a
        # malformed stub, so index directly and let it raise rather than masking it with a zero.
        return AggregateContext(
            tuple_id="",
            drug=drug_meta.get("drug", ""),
            cell_line=cell_meta.get("cell_name", ""),
            n_sig_genes_lfc05=int(y["n_sig_genes_lfc05"]),
            n_sig_genes_lfc10=int(y["n_sig_genes_lfc10"]),
            mean_abs_lfc=float(y["mean_abs_lfc"]),
            median_abs_lfc=float(y["median_abs_lfc"]),
            perturbation_strength_percentile=float(y["perturbation_strength_percentile"]),
            drug_metadata=drug_meta,
            cell_metadata=cell_meta,
        )

    return EvidenceTuple(
        tuple_id="",
        context_type=stub["context_type"],
        theta="",
        x="",
        y=y,
        c=stub.get("verdict") or "unresolved",
        drug_metadata=drug_meta,
        cell_metadata=cell_meta,
        gene_metadata=gene_meta,
    )


def _row_to_episode(row: dict, context_size: int, split: Split) -> TahoeEvidenceEpisode:
    target = TargetHypothesis(
        target_id="TARGET_1",
        drug=row["target_drug"],
        cell_line=row["target_cell"],
        gene=row["target_gene"],
        direction=row["target_direction"],
    )
    pool = list(row["context_pool"]) if row["context_pool"] is not None else []
    context = [_stub_to_context(stub) for stub in pool[:context_size]]
    return TahoeEvidenceEpisode(
        episode_id=row["episode_id"],
        split=Split(split),
        target=target,
        context=context,
        label=int(row["label"]),
        metadata={
            "lfc_threshold": float(row.get("lfc_threshold", 0.5) or 0.5),
            "target_drug_meta": dict(row.get("target_drug_meta") or {}),
            "target_cell_meta": dict(row.get("target_cell_meta") or {}),
            "target_gene_meta": dict(row.get("target_gene_meta") or {}),
        },
    )


@dataclass(frozen=True)
class RenderedEpisode:
    """An episode paired with its rendered user-prompt text.

    The shared unit every consumer builds on: the RL env wraps it into a verifiers record, the
    classification-head trainer into a (question, label) BCE example, and the eval/analysis
    scripts into their own rows.
    """

    episode: TahoeEvidenceEpisode
    question: str


def system_prompt_for(prompt_style: str, *, prior_note: bool = False) -> str:
    """System prompt matching a `prompt_style` ("direct" -> answer-only, else reasoning).

    The single source for the style->prompt mapping, shared by the RL env (`tahoe_meta.py`) and the
    classification-head trainer (`data_utils.py`) so both pick the same prompt for a given style;
    kept as a function rather than inlined to keep that choice in one place. `prior_note=True` inserts
    the shared PRIOR_NOTE (0.5-prior anchor) after the preamble, before the answer-format block.
    """
    if prompt_style == "direct":
        base = SYSTEM_PROMPT_DIRECT
    elif prompt_style == "reasoning":
        base = SYSTEM_PROMPT
    else:
        raise ValueError(f"unknown prompt_style: {prompt_style!r}")
    if prior_note:
        return _SYSTEM_PROMPT_PREAMBLE + PRIOR_NOTE + base[len(_SYSTEM_PROMPT_PREAMBLE) :]
    return base


def load_rendered_split(
    data_path: str,
    split: Split,
    *,
    context_size: int = 16,
    context_size_range: tuple[int, int] | None = None,
    context_seed: int = 0,
    prompt_mode: str = "compact",
    prompt_style: str = "reasoning",
    lfc_threshold: float = 0.5,
) -> list[RenderedEpisode]:
    """Load one split of a Tahoe collection and reconstruct + render every episode in it.

    The single split-load + render path shared by the RL env, the classification-head trainer, and
    the offline eval/analysis scripts: `load_dataset_split` -> `_row_to_episode` ->
    `PromptRenderer.render`. The split is the frame you load (train on `train`, select on `val`,
    hold out `test`), never a column to filter. Callers wrap the returned `RenderedEpisode`s into
    whatever record shape they need.

    Context length: each episode renders the first K stubs of its pool. By default K is the fixed
    `context_size`; set `context_size_range=(lo, hi)` to draw K per episode uniformly in `{lo..hi}`,
    deterministically keyed by `episode_id` (+ `context_seed`) — see `ibis_env_utils.context_size_for`.
    """
    context_size_range = validate_context_size_range(context_size_range)
    renderer = PromptRenderer(mode=prompt_mode, lfc_threshold=lfc_threshold, prompt_style=prompt_style)
    df = load_dataset_split(data_path, split)
    rendered = []
    for row in df.to_dict(orient="records"):
        k = context_size_for(row.get("episode_id"), context_size, context_size_range, context_seed)
        episode = _row_to_episode(row, k, split)
        rendered.append(RenderedEpisode(episode=episode, question=renderer.render(episode)))
    return rendered
