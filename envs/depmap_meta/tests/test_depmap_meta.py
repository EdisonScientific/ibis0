"""Unit tests for the DepMap Evidence Deliberation environment."""

import pandas as pd
import pytest
from depmap_meta import (
    _cell_block,
    _chronos_call,
    _drug_block,
    _gene_block,
    _pdep_call,
    build_record,
    compute_reward,
    load_environment,
    parse_prob,
    relevance_header,
    render_stub,
    render_user,
)


def _stub(gene="EGFR", cell="A549", ctype="same_gene_other_lineage", p_dep=0.8):
    return {
        "modality": "crispr",
        "context_type": ctype,
        "related": True,
        "gene": gene,
        "gene_metadata": {"symbol": gene},
        "cell": cell,
        "cell_metadata": {"cell_name": cell, "model_id": f"ACH-{cell}"},
        "y": {"chronos": -0.7, "p_dependency": p_dep},
    }


def _episode_row(eid, label=1, ood_group="iid", pool=None):
    """Minimal episode row sufficient for build_record / render_user."""
    return {
        "episode_id": eid,
        "target_gene": "KRAS",
        "target_cell": "CellLineX",
        "target_cell_id": "ACH-000123",
        "target_direction": "essential",
        "label": label,
        "ood_group": ood_group,
        "target_gene_meta": {"symbol": "KRAS", "name": "KRAS proto-oncogene"},
        "target_cell_meta": {"cell_name": "CellLineX", "model_id": "ACH-000123", "lineage": "Lung"},
        "context_pool": pool if pool is not None else [_stub(), _stub(cell="H1975"), _stub(cell="PC9")],
    }


def _write_splits(tmp_path, by_split):
    """Write one ``<split>.parquet`` per split into ``tmp_path`` and return the dir.

    The split is the file (no ``split`` column), so callers group rows by split here.
    """
    for split, rows in by_split.items():
        pd.DataFrame(rows).to_parquet(tmp_path / f"{split}.parquet")
    return str(tmp_path)


# ---- build_record ----


def test_build_record_question_describes_target():
    rec = build_record(_episode_row("e1", label=1), context_size=8)
    assert "KRAS" in rec["question"]
    assert "CellLineX" in rec["question"]
    assert rec["task"] == "depmap_evidence_deliberation"


def test_build_record_info_fields():
    rec = build_record(_episode_row("e1", label=0, ood_group="gene"), context_size=8)
    assert rec["info"] == {"episode_id": "e1", "label": 0, "ood_group": "gene", "context_size": 8}
    assert rec["answer"] == ""


def test_build_record_pool_sliced_to_context_size():
    # 3 stubs in the pool, context_size=2 -> only the first 2 render (nested prefixes).
    row = _episode_row("e1", pool=[_stub(cell="A"), _stub(cell="B"), _stub(cell="C")])
    rec = build_record(row, context_size=2)
    # render_user numbers stubs [1], [2], ...; with 2 stubs there is no "[3]"
    assert "[2]" in rec["question"]
    assert "[3]" not in rec["question"]


def test_build_record_handles_empty_pool():
    rec = build_record(_episode_row("e1", pool=[]), context_size=4)
    assert "KRAS" in rec["question"]
    assert rec["info"]["context_size"] == 4


def test_build_record_handles_none_pool():
    row = _episode_row("e1")
    row["context_pool"] = None
    rec = build_record(row, context_size=4)
    assert "KRAS" in rec["question"]


# ---- parse_prob + reward integration (reward math itself: ibis_env_utils/tests/test_reward.py) ----


def test_parse_prob_then_reward():
    p = parse_prob("<probability>0.75</probability>")
    assert compute_reward(p, 1) == pytest.approx(0.75)
    assert parse_prob("no probability here") is None


# ---- render_user ----


def test_render_user_renders_relevance_headers():
    text = render_user(_episode_row("e1"))
    assert "TARGET HYPOTHESIS" in text and "CONTEXT EVIDENCE" in text


# ---- load_environment (the env loads `train` and `val` from per-split files, never `test`) ----


def test_load_environment_loads_train_and_val_and_skips_test(tmp_path):
    path = _write_splits(
        tmp_path,
        {
            "train": [_episode_row("e1"), _episode_row("e2")],
            "val": [_episode_row("e3")],
            "test": [_episode_row("e4")],  # never loaded by the env
        },
    )
    env = load_environment(path, context_size=4)
    assert len(env.dataset) == 2
    assert len(env.eval_dataset) == 1


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
        from depmap_meta import SYSTEM_PROMPT_DIRECT, system_prompt_for

        assert system_prompt_for("direct") == SYSTEM_PROMPT_DIRECT
        assert "no reasoning, no explanation" in SYSTEM_PROMPT_DIRECT

    def test_for_reasoning_drops_no_reasoning_clause(self):
        from depmap_meta import SYSTEM_PROMPT, system_prompt_for

        assert system_prompt_for("reasoning") == SYSTEM_PROMPT
        assert "no reasoning, no explanation" not in SYSTEM_PROMPT
        assert SYSTEM_PROMPT.rstrip().endswith("<probability>0.XX</probability>")

    def test_variants_share_preamble(self):
        from depmap_meta import SYSTEM_PROMPT, SYSTEM_PROMPT_DIRECT

        preamble = SYSTEM_PROMPT_DIRECT.split("Respond with ONLY")[0]
        assert preamble and SYSTEM_PROMPT.startswith(preamble)

    def test_unknown_style_raises(self):
        from depmap_meta import system_prompt_for

        with pytest.raises(ValueError, match="prompt_style"):
            system_prompt_for("bogus")

    def test_load_environment_defaults_to_direct(self, tmp_path):
        from depmap_meta import SYSTEM_PROMPT_DIRECT

        path = _write_splits(tmp_path, {"train": [_episode_row("e1")], "val": [_episode_row("e2")]})
        env = load_environment(path, context_size=4)
        assert env.system_prompt == SYSTEM_PROMPT_DIRECT

    def test_load_environment_reasoning_style(self, tmp_path):
        from depmap_meta import SYSTEM_PROMPT

        path = _write_splits(tmp_path, {"train": [_episode_row("e1")], "val": [_episode_row("e2")]})
        env = load_environment(path, context_size=4, prompt_style="reasoning")
        assert env.system_prompt == SYSTEM_PROMPT


# ---- renderer internals ----


class TestChronosCall:
    """Chronos gene-effect score -> qualitative fitness call."""

    def test_none_is_no_data(self):
        assert _chronos_call(None) == "no data"

    def test_strongly_negative_is_substantial(self):
        assert _chronos_call(-0.5) == "substantial fitness cost"
        assert _chronos_call(-1.2) == "substantial fitness cost"

    def test_mildly_negative_is_mild(self):
        assert _chronos_call(-0.1) == "mild fitness reduction"
        assert _chronos_call(-0.3) == "mild fitness reduction"

    def test_near_zero_is_no_clear_effect(self):
        assert _chronos_call(0.0) == "no clear fitness effect"
        assert _chronos_call(0.25) == "no clear fitness effect"

    def test_positive_above_threshold_is_suppressor(self):
        assert _chronos_call(0.26) == "knockout speeds growth — suppressor"
        assert _chronos_call(1.0) == "knockout speeds growth — suppressor"


class TestPdepCall:
    """Calibrated P(dependency) -> qualitative dependency call."""

    def test_none_is_no_data(self):
        assert _pdep_call(None) == "no data"

    def test_at_or_above_half_is_likely(self):
        assert _pdep_call(0.5) == "likely a true dependency"
        assert _pdep_call(0.95) == "likely a true dependency"

    def test_mid_range_is_unlikely(self):
        assert _pdep_call(0.1) == "unlikely a true dependency"
        assert _pdep_call(0.49) == "unlikely a true dependency"

    def test_low_is_almost_certainly_not(self):
        assert _pdep_call(0.09) == "almost certainly not a dependency"
        assert _pdep_call(0.0) == "almost certainly not a dependency"


class TestRelevanceHeader:
    """Per-stub relevance header derived from context_type."""

    def test_known_context_types(self):
        assert relevance_header({"context_type": "same_gene_same_lineage"}) == (
            "target gene knocked out in a different cell line of the same lineage"
        )
        assert relevance_header({"context_type": "molecular_mutation"}) == (
            "somatic mutation status of the target gene in the target cell line"
        )
        assert relevance_header({"context_type": "control"}) == "an unrelated gene knocked out in the target cell line"

    def test_pharmacology_default(self):
        assert relevance_header({"context_type": "pharmacology"}) == (
            "a drug targeting the target gene, profiled in the target cell line"
        )

    def test_pharmacology_via_pathway_gene_special_case(self):
        s = {"context_type": "pharmacology", "via_pathway_gene": "EGFR"}
        assert relevance_header(s) == (
            "a drug targeting a pathway partner of the target gene, profiled in the target cell line"
        )

    def test_unknown_context_type_falls_back_to_value(self):
        assert relevance_header({"context_type": "novel_relation"}) == "novel_relation"

    def test_missing_context_type_falls_back_to_evidence(self):
        assert relevance_header({}) == "evidence"


class TestSystemPromptTierScaffolding:
    """The system prompt must NOT enumerate the context types up front — inferring each stub's
    relevance is part of the task (as in gwas_meta). Per-stub relation headers
    are kept."""

    def test_no_context_type_enumeration_in_system_prompt(self):
        from depmap_meta import SYSTEM_PROMPT

        assert "CRISPR knockouts of related genes/cell lines" not in SYSTEM_PROMPT
        assert "molecular state, and drug sensitivity" not in SYSTEM_PROMPT
        # the generic CONTEXT description (and the per-stub labeling it refers to) is retained
        assert "each labeled by its relation to the target" in SYSTEM_PROMPT

    def test_per_stub_headers_still_present(self):
        text = render_user(_episode_row("e1"))
        assert relevance_header({"context_type": "same_gene_other_lineage"}) in text


class TestGeneBlock:
    """Gene metadata block formatting."""

    def test_empty_returns_blank(self):
        assert _gene_block(None) == ""
        assert _gene_block({}) == ""

    def test_symbol_and_name_only(self):
        block = _gene_block({"symbol": "BRCA1", "name": "BRCA1 DNA repair associated"})
        assert block == "      Gene: BRCA1 — BRCA1 DNA repair associated"

    def test_name_absent_falls_back_to_symbol(self):
        block = _gene_block({"symbol": "KRAS", "symbol_alias": "KRAS"})
        assert block == "      Gene: KRAS — KRAS"

    def test_family_and_ensembl_appended(self):
        block = _gene_block(
            {
                "symbol": "EGFR",
                "name": "epidermal growth factor receptor",
                "family": "receptor tyrosine kinase",
                "ensembl_id": "ENSG00000146648",
            }
        )
        assert block == (
            "      Gene: EGFR — epidermal growth factor receptor "
            "(family: receptor tyrosine kinase; Ensembl ENSG00000146648)"
        )

    def test_only_family_present(self):
        block = _gene_block({"symbol": "TP53", "name": "tumor protein p53", "family": "p53 family"})
        assert block == "      Gene: TP53 — tumor protein p53 (family: p53 family)"


class TestCellBlock:
    """Cell line metadata block formatting."""

    def test_empty_returns_blank(self):
        assert _cell_block(None) == ""
        assert _cell_block({}) == ""

    def test_minimal_fields(self):
        block = _cell_block({"cell_name": "A549", "model_id": "ACH-000681", "lineage": "lung"})
        assert block == "      Cell line: A549 (DepMap ACH-000681) — lung"

    def test_disease_subtype_and_demographics(self):
        block = _cell_block(
            {
                "cell_name": "A549",
                "model_id": "ACH-000681",
                "lineage": "lung",
                "disease": "non-small cell lung cancer",
                "subtype": "adenocarcinoma",
                "sex": "male",
                "primary_metastasis": "primary",
            }
        )
        assert block == (
            "      Cell line: A549 (DepMap ACH-000681) — lung "
            "/ non-small cell lung cancer / adenocarcinoma; male; primary"
        )

    def test_driver_mutations_appended(self):
        block = _cell_block(
            {
                "cell_name": "A549",
                "model_id": "ACH-000681",
                "lineage": "lung",
                "driver_mutations": ["KRAS G12S", "STK11 Q37*"],
            }
        )
        assert block.endswith("\n        Driver mutations: KRAS G12S, STK11 Q37*")

    @pytest.mark.parametrize("driver_mutations", [[], None])
    def test_falsy_driver_mutations_omitted(self, driver_mutations):
        block = _cell_block(
            {"cell_name": "A549", "model_id": "ACH-000681", "lineage": "lung", "driver_mutations": driver_mutations}
        )
        assert "Driver mutations" not in block


class TestDrugBlock:
    """Drug metadata block formatting."""

    def test_empty_returns_blank(self):
        assert _drug_block(None) == ""
        assert _drug_block({}) == ""

    def test_name_only(self):
        assert _drug_block({"drug": "erlotinib"}) == "      Drug: erlotinib"

    def test_targets_moa_chembl(self):
        block = _drug_block(
            {
                "drug": "erlotinib",
                "targets": "EGFR",
                "moa": "EGFR inhibitor",
                "chembl_id": "CHEMBL553",
            }
        )
        assert block == ("      Drug: erlotinib\n        targets: EGFR; MOA: EGFR inhibitor; ChEMBL CHEMBL553")

    def test_smiles_appended(self):
        block = _drug_block({"drug": "erlotinib", "smiles": "C#Cc1cccc(c1)Nc1ncnc2cc(OCCOC)c(OCCOC)cc12"})
        assert block == ("      Drug: erlotinib\n        SMILES: C#Cc1cccc(c1)Nc1ncnc2cc(OCCOC)c(OCCOC)cc12")


class TestRenderStubCrispr:
    def test_crispr_block(self):
        stub = {
            "modality": "crispr",
            "gene": "PLK1",
            "cell": "A549",
            "gene_metadata": {"symbol": "PLK1", "name": "polo like kinase 1"},
            "cell_metadata": {"cell_name": "A549", "model_id": "ACH-000681", "lineage": "lung"},
            "y": {"chronos": -1.05, "p_dependency": 0.98},
        }
        out = render_stub(stub)
        assert out.startswith("CRISPR knockout of PLK1 in A549")
        assert "Chronos gene-effect  -1.05  (substantial fitness cost)" in out
        assert "P(dependency)        0.98   (likely a true dependency)" in out
        assert "Gene: PLK1" in out
        assert "Cell line: A549" in out

    def test_crispr_non_dependency(self):
        stub = {
            "modality": "crispr",
            "gene": "OR2T1",
            "cell": "A549",
            "y": {"chronos": 0.05, "p_dependency": 0.02},
        }
        out = render_stub(stub)
        assert "+0.05  (no clear fitness effect)" in out
        assert "0.02   (almost certainly not a dependency)" in out


class TestRenderStubExpression:
    def _stub(self, z, log2_tpm=6.4):
        return {
            "modality": "expression",
            "gene": "MET",
            "cell": "A549",
            "y": {"lineage_z": z, "log2_tpm": log2_tpm},
        }

    def test_high_z_significantly_high(self):
        out = render_stub(self._stub(2.3))
        assert out.startswith("Baseline mRNA expression of MET in A549")
        assert "z = +2.3 vs lineage (significantly high)" in out

    def test_low_z_significantly_low(self):
        out = render_stub(self._stub(-2.1))
        assert "z = -2.1 vs lineage (significantly low)" in out

    def test_neutral_z_no_annotation(self):
        out = render_stub(self._stub(0.4))
        assert "z = +0.4 vs lineage" in out
        assert "significantly" not in out

    def test_log2_tpm_rendered(self):
        out = render_stub(self._stub(0.4, log2_tpm=6.4))
        assert "log2(TPM+1) = 6.4" in out


class TestRenderStubCopyNumber:
    def test_copy_number_block(self):
        stub = {
            "modality": "copy_number",
            "gene": "MYC",
            "cell": "A549",
            "y": {"relative_cn": 2.8, "call": "amplification"},
        }
        out = render_stub(stub)
        assert out.startswith("Copy number of the MYC locus in A549")
        assert "relative copy number = 2.8 (amplification)" in out


class TestRenderStubMutation:
    def test_wild_type(self):
        stub = {"modality": "mutation", "gene": "TP53", "cell": "A549", "y": {"status": "wild-type"}}
        out = render_stub(stub)
        assert out.startswith("Somatic mutation status of TP53 in A549")
        assert "Result: wild-type" in out

    def test_mutated_plain(self):
        stub = {
            "modality": "mutation",
            "gene": "KRAS",
            "cell": "A549",
            "y": {"status": "mutated", "protein_change": "p.G12S"},
        }
        out = render_stub(stub)
        assert "Result: mutation p.G12S" in out

    def test_mutated_hotspot_and_damaging(self):
        stub = {
            "modality": "mutation",
            "gene": "KRAS",
            "cell": "A549",
            "y": {"status": "mutated", "hotspot": True, "damaging": True, "protein_change": "p.G12S"},
        }
        out = render_stub(stub)
        assert "Result: hotspot damaging mutation p.G12S" in out

    def test_mutated_no_protein_change(self):
        stub = {"modality": "mutation", "gene": "STK11", "cell": "A549", "y": {"status": "mutated"}}
        out = render_stub(stub)
        assert "Result: mutation" in out


class TestRenderStubPharmacology:
    def _stub(self, **y):
        return {
            "modality": "pharmacology",
            "gene": "EGFR",
            "cell": "A549",
            "drug_metadata": {"drug": "erlotinib", "targets": "EGFR", "moa": "EGFR inhibitor"},
            "y": y,
        }

    def test_sensitive(self):
        out = render_stub(self._stub(sensitive=True, viability_log2fc=-1.4))
        assert out.startswith("Pharmacological sensitivity of A549 to erlotinib")
        assert "PRISM viability log2FC = -1.40 (sensitive; more negative = more cell killing)" in out
        assert "Drug: erlotinib" in out

    def test_resistant(self):
        out = render_stub(self._stub(sensitive=False, viability_log2fc=-0.1))
        assert "-0.10 (resistant;" in out

    def test_modest_sensitivity(self):
        out = render_stub(self._stub(sensitive=False, viability_log2fc=-0.8))
        assert "-0.80 (modest sensitivity;" in out


class TestRenderStubUnknownModality:
    def test_fallback(self):
        assert render_stub({"modality": "proteomics"}) == "proteomics stub"


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

    def test_no_probability_after_think_returns_none(self):
        assert parse_prob("<probability>0.42</probability></think> no tag here") is None

    def test_multiple_tags_uses_last(self):
        assert parse_prob("<probability>0.10</probability> then <probability>0.80</probability>") == 0.80


def _gene_meta(symbol="PLK1", name="polo like kinase 1"):
    return {"symbol": symbol, "name": name}


def _cell_meta(cell_name="A549"):
    return {"cell_name": cell_name, "model_id": "ACH-000681", "lineage": "lung"}


def _crispr_stub(context_type, gene, chronos=-0.9, p_dependency=0.95):
    return {
        "modality": "crispr",
        "context_type": context_type,
        "gene": gene,
        "cell": "A549",
        "y": {"chronos": chronos, "p_dependency": p_dependency},
    }


class TestRenderUser:
    def test_sections_and_target(self):
        row = {
            "target_gene": "PLK1",
            "target_gene_meta": _gene_meta(),
            "target_cell_meta": _cell_meta(),
            "context_pool": [
                _crispr_stub("same_gene_other_lineage", "PLK1"),
                _crispr_stub("control", "OR2T1", chronos=0.0, p_dependency=0.02),
            ],
        }
        out = render_user(row)

        assert "CONTEXT EVIDENCE" in out
        assert "TARGET HYPOTHESIS" in out
        # CONTEXT section precedes the TARGET HYPOTHESIS section.
        assert out.index("CONTEXT EVIDENCE") < out.index("TARGET HYPOTHESIS")

        assert "[1] target gene knocked out in a cell line of a different lineage" in out
        assert "[2] an unrelated gene knocked out in the target cell line" in out

        assert "Gene PLK1 is a dependency" in out
        assert "in cell line A549" in out
        assert "Chronos gene-effect <= -0.5 AND P(dependency) >= 0.5" in out
        assert "<probability>0.XX</probability>" in out

    def test_target_cell_falls_back_to_target_cell_field(self):
        row = {
            "target_gene": "PLK1",
            "target_gene_meta": None,
            "target_cell_meta": None,
            "target_cell": "HCT116",
            "context_pool": [],
        }
        out = render_user(row)
        assert "in cell line HCT116" in out

    def test_empty_context_pool_still_renders_target(self):
        row = {
            "target_gene": "MYC",
            "target_gene_meta": _gene_meta("MYC", "MYC proto-oncogene"),
            "target_cell_meta": _cell_meta(),
            "context_pool": [],
        }
        out = render_user(row)
        assert "Gene MYC is a dependency" in out
        assert "Gene: MYC — MYC proto-oncogene" in out
