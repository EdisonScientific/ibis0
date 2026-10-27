"""Unit tests for the GWAS Catalog Evidence Deliberation environment."""

import numpy as np
import pandas as pd
import pytest
from gwas_meta import (
    _gene_context_block,
    _hypothesis_sentence,
    _pval_call,
    _result_block,
    _snp_block,
    _study_block,
    build_record,
    compute_reward,
    load_environment,
    parse_prob,
    relevance_header,
    render_stub,
    render_user,
)


def _stub(snp="rs7901695", gene="TCF7L2", trait="type 2 diabetes mellitus", ctype="same_gene_same_trait", p=8e-62):
    return {
        "modality": "association",
        "context_type": ctype,
        "related": True,
        "snp": snp,
        "gene": gene,
        "trait": trait,
        "snp_metadata": {
            "rsid": snp,
            "chr": "10",
            "pos": "112994329",
            "risk_allele": "C",
            "raf": 0.41,
            "functional_class": "intron_variant",
            "intergenic": False,
            "distance_to_target": 4261,
        },
        "gene_metadata": {
            "symbol": gene,
            "ensembl_id": "ENSG00000148737",
            "reactome_pathways": ["WNT signaling"],
            "gtex_top_tissues": "Artery-Tibial 38.8 TPM",
        },
        "trait_metadata": {"trait": trait, "efo_parent": "Metabolic disorder"},
        "study_metadata": {
            "accession": "GCST010118",
            "ancestry": "East Asian",
            "sample_size": 433540,
            "platform": "Illumina",
        },
        "y": {
            "p_value": p,
            "or_beta": 1.32,
            "ci_text": "[1.29-1.35]",
            "direction": 1,
            "direction_phrase": "C allele increases type 2 diabetes mellitus",
        },
    }


def _episode_row(eid, label=1, ood_group="iid", pool=None):
    """Minimal episode row sufficient for build_record / render_user."""
    return {
        "episode_id": eid,
        "target_snp": "rs7903146",
        "target_trait": "type 2 diabetes mellitus",
        "target_gene": "TCF7L2",
        "target_allele": "T",
        "target_direction": 1,
        "target_efo_parent": "Metabolic disorder",
        "label": label,
        "negative_type": None,
        "sufficiency_tier": "rich",
        "ood_group": ood_group,
        "ground_truth": {"p_value": 3e-11, "or_beta": 1.40, "allele": "T", "direction": 1, "study": "GCSTX"},
        "target_snp_meta": {
            "rsid": "rs7903146",
            "chr": "10",
            "pos": "112998590",
            "risk_allele": "T",
            "raf": 0.3,
            "functional_class": "intron_variant",
            "intergenic": False,
        },
        "target_gene_meta": {
            "symbol": "TCF7L2",
            "ensembl_id": "ENSG00000148737",
            "reactome_pathways": ["WNT signaling"],
            "gtex_top_tissues": "Artery-Tibial 38.8 TPM",
        },
        "target_trait_meta": {"trait": "type 2 diabetes mellitus", "efo_parent": "Metabolic disorder"},
        "context_pool": pool if pool is not None else [_stub(), _stub(snp="rs117229942"), _stub(snp="rs11196205")],
    }


def _write_splits(tmp_path, by_split):
    """Write one ``<split>.parquet`` per split into ``tmp_path`` and return the dir."""
    for split, rows in by_split.items():
        pd.DataFrame(rows).to_parquet(tmp_path / f"{split}.parquet")
    return str(tmp_path)


# ---- build_record ----


def test_build_record_question_describes_target():
    rec = build_record(_episode_row("e1", label=1), context_size=8)
    assert "rs7903146" in rec["question"]
    assert "type 2 diabetes mellitus" in rec["question"]
    assert rec["task"] == "gwas_evidence_deliberation"


def test_build_record_info_fields():
    rec = build_record(_episode_row("e1", label=0, ood_group="gene"), context_size=8)
    assert rec["info"] == {"episode_id": "e1", "label": 0, "ood_group": "gene", "context_size": 8}
    assert rec["answer"] == ""


def test_build_record_pool_sliced_to_context_size():
    row = _episode_row("e1", pool=[_stub(snp="rsA"), _stub(snp="rsB"), _stub(snp="rsC")])
    rec = build_record(row, context_size=2)
    assert "[2]" in rec["question"]
    assert "[3]" not in rec["question"]


def test_build_record_handles_empty_pool():
    rec = build_record(_episode_row("e1", pool=[]), context_size=4)
    assert "rs7903146" in rec["question"]
    assert rec["info"]["context_size"] == 4


def test_build_record_handles_none_pool():
    row = _episode_row("e1")
    row["context_pool"] = None
    rec = build_record(row, context_size=4)
    assert "rs7903146" in rec["question"]


# ---- parse_prob + reward integration ----


def test_parse_prob_then_reward():
    p = parse_prob("<probability>0.75</probability>")
    assert compute_reward(p, 1) == pytest.approx(0.75)
    assert parse_prob("no probability here") is None


# ---- render_user ----


def test_render_user_sections_and_target():
    row = _episode_row(
        "e1",
        pool=[
            _stub(ctype="same_gene_same_trait"),
            _stub(snp="rsZ", gene="MCM6", trait="lactose intolerance", ctype="control"),
        ],
    )
    out = render_user(row)
    assert "CONTEXT EVIDENCE" in out and "TARGET HYPOTHESIS" in out
    assert out.index("CONTEXT EVIDENCE") < out.index("TARGET HYPOTHESIS")
    assert "[1] a different SNP in the target gene, associated with the target trait" in out
    assert "[2] an unrelated SNP and trait (different gene, different EFO parent)" in out
    assert "SNP rs7903146 (T allele) increases type 2 diabetes mellitus." in out
    assert "p < 5e-8" in out
    assert "EFO parent: Metabolic disorder" in out
    assert "<probability>0.XX</probability>" in out


def test_render_user_target_falls_back_to_associated_when_no_direction():
    row = _episode_row("e1")
    row["target_direction"] = None
    out = render_user(row)
    assert "is associated with type 2 diabetes mellitus." in out


def test_render_user_empty_pool_still_renders_target():
    out = render_user(_episode_row("e1", pool=[]))
    assert "SNP rs7903146 (T allele) increases" in out


# ---- load_environment (loads train + val from per-split files, never test) ----


def test_load_environment_loads_train_and_val_and_skips_test(tmp_path):
    path = _write_splits(
        tmp_path,
        {
            "train": [_episode_row("e1"), _episode_row("e2")],
            "val": [_episode_row("e3")],
            "test": [_episode_row("e4")],
        },
    )
    env = load_environment(path, context_size=4)
    assert len(env.dataset) == 2
    assert len(env.eval_dataset) == 1


def test_load_environment_renders_empty_reactome_from_parquet(tmp_path):
    """Regression: reactome_pathways round-trips through parquet as a numpy array; an empty one (a gene
    with no Reactome pathway) must render without raising --- the failure mode seen on the real build."""
    empty = _episode_row("e1")
    empty["target_gene_meta"] = {
        "symbol": "X",
        "ensembl_id": "ENSG0",
        "reactome_pathways": [],
        "gtex_top_tissues": None,
    }
    for s in empty["context_pool"]:
        s["gene_metadata"]["reactome_pathways"] = []
        s["gene_metadata"]["gtex_top_tissues"] = None
    path = _write_splits(tmp_path, {"train": [empty], "val": [_episode_row("e2")]})
    env = load_environment(path, context_size=4)
    assert "rs7903146" in env.dataset[0]["prompt"][-1]["content"]  # rendered without error


def test_load_environment_missing_train_raises(tmp_path):
    path = _write_splits(tmp_path, {"val": [_episode_row("e1")], "test": [_episode_row("e2")]})
    with pytest.raises(FileNotFoundError, match="train"):
        load_environment(path)


def test_load_environment_missing_val_raises(tmp_path):
    path = _write_splits(tmp_path, {"train": [_episode_row("e1")]})
    with pytest.raises(FileNotFoundError, match="val"):
        load_environment(path)


def test_load_environment_empty_train_raises(tmp_path):
    path = _write_splits(tmp_path, {"train": [], "val": [_episode_row("e1")]})
    with pytest.raises(ValueError, match="train"):
        load_environment(path)


class TestPromptStyle:
    """direct vs. reasoning system-prompt selection (shared via ibis_env_utils.select_system_prompt).

    Thinking-mode runs (`enable_thinking=true`) use the reasoning prompt; direct runs use the
    answer-only prompt. `prompt_style="direct"` is the default.
    """

    def test_for_direct_is_answer_only(self):
        from gwas_meta import SYSTEM_PROMPT_DIRECT, system_prompt_for

        assert system_prompt_for("direct") == SYSTEM_PROMPT_DIRECT
        assert "no reasoning, no explanation" in SYSTEM_PROMPT_DIRECT

    def test_for_reasoning_drops_no_reasoning_clause(self):
        from gwas_meta import SYSTEM_PROMPT, system_prompt_for

        assert system_prompt_for("reasoning") == SYSTEM_PROMPT
        assert "no reasoning, no explanation" not in SYSTEM_PROMPT
        assert SYSTEM_PROMPT.rstrip().endswith("<probability>0.XX</probability>")

    def test_variants_share_preamble(self):
        from gwas_meta import SYSTEM_PROMPT, SYSTEM_PROMPT_DIRECT

        preamble = SYSTEM_PROMPT_DIRECT.split("Respond with ONLY")[0]
        assert preamble and SYSTEM_PROMPT.startswith(preamble)

    def test_unknown_style_raises(self):
        from gwas_meta import system_prompt_for

        with pytest.raises(ValueError, match="prompt_style"):
            system_prompt_for("bogus")

    def test_load_environment_defaults_to_direct(self, tmp_path):
        from gwas_meta import SYSTEM_PROMPT_DIRECT

        path = _write_splits(tmp_path, {"train": [_episode_row("e1")], "val": [_episode_row("e2")]})
        env = load_environment(path, context_size=4)
        assert env.system_prompt == SYSTEM_PROMPT_DIRECT

    def test_load_environment_reasoning_style(self, tmp_path):
        from gwas_meta import SYSTEM_PROMPT

        path = _write_splits(tmp_path, {"train": [_episode_row("e1")], "val": [_episode_row("e2")]})
        env = load_environment(path, context_size=4, prompt_style="reasoning")
        assert env.system_prompt == SYSTEM_PROMPT


# ---- renderer internals ----


class TestPvalCall:
    def test_none_is_no_data(self):
        assert _pval_call(None) == "no data"

    def test_below_genome_wide(self):
        assert _pval_call(4e-11) == "genome-wide significant"
        assert _pval_call(4.9e-8) == "genome-wide significant"

    def test_suggestive(self):
        assert _pval_call(5e-8) == "suggestive, below the genome-wide bar"
        assert _pval_call(9e-6) == "suggestive, below the genome-wide bar"

    def test_not_significant(self):
        assert _pval_call(1e-5) == "not significant"
        assert _pval_call(0.2) == "not significant"


class TestRelevanceHeader:
    def test_known_context_types(self):
        assert relevance_header({"context_type": "same_gene_same_trait"}) == (
            "a different SNP in the target gene, associated with the target trait"
        )
        assert relevance_header({"context_type": "same_snp_unrelated_trait"}) == (
            "the target SNP, associated with a trait in a different EFO parent"
        )
        assert relevance_header({"context_type": "control"}) == (
            "an unrelated SNP and trait (different gene, different EFO parent)"
        )

    def test_unknown_falls_back_to_value(self):
        assert relevance_header({"context_type": "novel_relation"}) == "novel_relation"

    def test_missing_falls_back_to_evidence(self):
        assert relevance_header({}) == "evidence"


class TestSystemPromptTierScaffolding:
    """The system prompt must NOT enumerate the context tiers up front — inferring each stub's
    relevance is part of the task. Per-stub relation headers are kept."""

    def test_no_tier_enumeration_in_system_prompt(self):
        from gwas_meta import SYSTEM_PROMPT

        assert "and unrelated controls" not in SYSTEM_PROMPT
        assert "pathway-neighbor genes," not in SYSTEM_PROMPT
        assert "the same SNP tested for other traits" not in SYSTEM_PROMPT
        # the generic CONTEXT description (and the per-stub labeling it refers to) is retained
        assert "each labeled by its relation to the target" in SYSTEM_PROMPT

    def test_per_stub_headers_still_present(self):
        row = _episode_row("e1", pool=[_stub(ctype="same_gene_same_trait")])
        out = render_user(row)
        assert "[1] a different SNP in the target gene, associated with the target trait" in out


class TestSnpBlock:
    def test_full(self):
        block = _snp_block(_stub())
        assert "SNP: rs7901695 — chr10:112994329" in block
        assert "Distance to target: 4,261 bp" in block
        assert "Risk allele: C" in block
        assert "RAF: 0.41" in block
        assert "Mapped gene: TCF7L2" in block
        assert "Functional class: intron_variant" in block

    def test_intergenic_flag(self):
        s = _stub()
        s["snp_metadata"]["intergenic"] = True
        assert "Intergenic: yes" in _snp_block(s)

    def test_no_distance_when_absent(self):
        s = _stub()
        s["snp_metadata"]["distance_to_target"] = None
        assert "Distance to target" not in _snp_block(s)


class TestGeneContextBlock:
    def test_empty_returns_blank(self):
        assert _gene_context_block(None) == ""
        assert _gene_context_block({"symbol": "X"}) == ""  # no pathways/gtex -> no block

    def test_pathways_and_gtex(self):
        block = _gene_context_block(_stub()["gene_metadata"])
        assert "Gene context:" in block
        assert "Reactome pathways: WNT signaling" in block
        assert "GTEx expression: Artery-Tibial 38.8 TPM" in block

    def test_empty_numpy_pathways_from_parquet(self):
        # regression: parquet round-trips reactome_pathways as a numpy array; an EMPTY one must not
        # raise "truth value of an empty array is ambiguous" (real genes often have no Reactome pathway)
        gm = {"symbol": "X", "reactome_pathways": np.array([], dtype=object), "gtex_top_tissues": None}
        assert _gene_context_block(gm) == ""

    def test_numpy_array_pathways_rendered(self):
        gm = {"symbol": "X", "reactome_pathways": np.array(["WNT signaling", "GLP-1"], dtype=object)}
        assert "Reactome pathways: WNT signaling, GLP-1" in _gene_context_block(gm)

    def test_nan_gtex_not_rendered(self):
        # parquet null gtex comes back as NaN (float), which is truthy — must not render "nan"
        gm = {"symbol": "X", "reactome_pathways": ["WNT"], "gtex_top_tissues": float("nan")}
        assert "GTEx" not in _gene_context_block(gm)


class TestStudyBlock:
    def test_full(self):
        block = _study_block(_stub()["study_metadata"])
        assert "Study: GCST010118" in block
        assert "Ancestry: East Asian" in block
        assert "Sample size: 433,540" in block
        assert "Platform: Illumina" in block

    def test_empty_without_accession(self):
        assert _study_block({"ancestry": "European"}) == ""


class TestResultBlock:
    def test_significant_with_direction(self):
        block = _result_block(_stub()["y"])
        assert "p = 8.0e-62 (genome-wide significant); OR/beta = 1.32" in block
        assert "95% CI: [1.29-1.35]" in block
        assert "Direction: C allele increases type 2 diabetes mellitus" in block

    def test_missing_or_beta(self):
        block = _result_block({"p_value": 2e-11, "or_beta": None})
        assert "OR/beta = N/A" in block


class TestRenderStub:
    def test_header_and_body(self):
        out = render_stub(_stub())
        assert out.startswith("rs7901695 for type 2 diabetes mellitus")
        assert "SNP: rs7901695" in out
        assert "Study: GCST010118" in out
        assert "Result: p = 8.0e-62" in out


class TestHypothesisSentence:
    def test_increase(self):
        assert _hypothesis_sentence(_episode_row("e")) == (
            "SNP rs7903146 (T allele) increases type 2 diabetes mellitus."
        )

    def test_decrease(self):
        row = _episode_row("e")
        row["target_direction"] = -1
        assert "decreases type 2 diabetes mellitus." in _hypothesis_sentence(row)

    def test_no_direction(self):
        row = _episode_row("e")
        row["target_direction"] = None
        assert "is associated with type 2 diabetes mellitus." in _hypothesis_sentence(row)

    def test_allele_from_meta_when_target_allele_missing(self):
        row = _episode_row("e")
        row["target_allele"] = None  # falls back to target_snp_meta.risk_allele
        assert "(T allele)" in _hypothesis_sentence(row)


class TestParseProb:
    def test_plain_tag(self):
        assert parse_prob("<probability>0.73</probability>") == 0.73

    def test_clamps_above_one(self):
        assert parse_prob("<probability>1.5</probability>") == 1.0

    def test_strips_text_before_last_think_close(self):
        text = "<think><probability>0.10</probability> reconsidering</think>\n<probability>0.90</probability>"
        assert parse_prob(text) == 0.90

    def test_no_match_returns_none(self):
        assert parse_prob("I am not sure about this hypothesis.") is None

    def test_multiple_tags_uses_last(self):
        assert parse_prob("<probability>0.10</probability> then <probability>0.80</probability>") == 0.80
