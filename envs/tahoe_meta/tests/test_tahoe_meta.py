"""Unit tests for the Tahoe Evidence Deliberation environment (single-target, shared pipeline)."""

import pandas as pd
import pytest
from episodes import (
    AggregateContext,
    EvidenceTuple,
    PromptRenderer,
    Split,
    TahoeEvidenceEpisode,
    TargetHypothesis,
    compute_verdict,
    render_user,
)
from ibis_env_utils import compute_accuracy, compute_reward, parse_prob
from tahoe_meta import build_record, load_environment

# ---- compute_verdict (label derivation) ----


def test_supported_upregulation():
    assert compute_verdict(log2fc=1.5, padj=0.01, claimed_direction="upregulates", lfc_threshold=0.5) == "supported"


def test_supported_downregulation():
    assert compute_verdict(log2fc=-1.5, padj=0.001, claimed_direction="downregulates", lfc_threshold=0.5) == "supported"


def test_refuted_wrong_direction():
    assert compute_verdict(log2fc=-1.5, padj=0.01, claimed_direction="upregulates", lfc_threshold=0.5) == "refuted"


def test_unresolved_not_significant_padj():
    assert compute_verdict(log2fc=1.5, padj=0.10, claimed_direction="upregulates", lfc_threshold=0.5) == "unresolved"


def test_unresolved_small_lfc():
    assert compute_verdict(log2fc=0.3, padj=0.01, claimed_direction="upregulates", lfc_threshold=0.5) == "unresolved"


def test_unresolved_missing_data():
    assert compute_verdict(None, 0.01, "upregulates") == "unresolved"
    assert compute_verdict(1.0, None, "upregulates") == "unresolved"


def test_stricter_threshold():
    assert compute_verdict(log2fc=0.8, padj=0.01, claimed_direction="upregulates", lfc_threshold=1.0) == "unresolved"
    assert compute_verdict(log2fc=1.5, padj=0.01, claimed_direction="upregulates", lfc_threshold=1.0) == "supported"


# ---- schema ----


def test_evidence_tuple_has_single_gene():
    evidence = EvidenceTuple(
        tuple_id="test123",
        context_type="same_drug_diff_cell",
        theta="",
        x="",
        y={"gene": "ABC", "log2FoldChange": 1.5, "padj": 0.01},
        c="supported",
        drug_metadata={},
        cell_metadata={},
    )
    assert isinstance(evidence.y["gene"], str)
    assert evidence.y["gene"] == "ABC"


def test_aggregate_has_summary_stats():
    agg = AggregateContext(
        tuple_id="agg",
        drug="D",
        cell_line="C",
        n_sig_genes_lfc05=100,
        n_sig_genes_lfc10=30,
        mean_abs_lfc=0.7,
        median_abs_lfc=0.5,
        perturbation_strength_percentile=60.0,
    )
    assert agg.n_sig_genes_lfc05 == 100
    assert not hasattr(agg, "log2FoldChange")


# ---- prompt rendering (shared layout, single-target scalar label) ----


def _episode(label=1, context=None):
    return TahoeEvidenceEpisode(
        episode_id="test",
        split=Split.TRAIN,
        target=TargetHypothesis("TARGET_1", "DrugX", "CellY", "GeneZ", "upregulates"),
        context=context or [],
        label=label,
    )


def test_episode_carries_scalar_label():
    ep = _episode(label=0)
    assert ep.label == 0
    assert isinstance(ep.target, TargetHypothesis)


def test_compact_prompt_shared_layout_and_target():
    prompt = PromptRenderer(mode="compact").render(_episode())
    assert prompt.startswith("CONTEXT EVIDENCE\n================")
    assert "TARGET HYPOTHESIS\n=================" in prompt
    assert "Drug DrugX upregulates gene GeneZ in cell line CellY." in prompt
    # style-independent footer from the shared skeleton
    assert prompt.endswith(
        "Respond using the format specified in the system message:\n  <probability>0.XX</probability>"
    )
    # empty-context notice
    assert "no prior evidence provided" in prompt
    # internal target_id is not surfaced
    assert "TARGET_1" not in prompt


def test_debug_prompt_shows_ground_truth():
    out = PromptRenderer(mode="debug").render(_episode(label=1))
    assert "GROUND TRUTH" in out
    assert "supported" in out


def test_module_render_user_matches_prompt_renderer():
    """The RL/classifier `render_user(row)` and the analysis `PromptRenderer` are one renderer."""
    ctx = EvidenceTuple(
        tuple_id="1",
        context_type="same_drug_diff_cell",
        theta="",
        x="",
        y={"gene": "GeneZ", "log2FoldChange": 1.0, "padj": 0.01},
        c="supported",
        drug_metadata={"drug": "DrugA"},
        cell_metadata={"cell_name": "CellX"},
        gene_metadata={"gene_symbol": "GeneZ"},
    )
    ep = _episode(context=[ctx])
    row = {
        "context_pool": [
            {
                "kind": "evidence",
                "context_type": "same_drug_diff_cell",
                "drug_metadata": {"drug": "DrugA"},
                "cell_metadata": {"cell_name": "CellX"},
                "gene_metadata": {"gene_symbol": "GeneZ"},
                "y": {"gene": "GeneZ", "log2FoldChange": 1.0, "padj": 0.01},
            }
        ],
        "target_drug": "DrugX",
        "target_cell": "CellY",
        "target_gene": "GeneZ",
        "target_direction": "upregulates",
    }
    assert render_user(row) == PromptRenderer(mode="compact").render(ep)


# ---- parsing (single <probability> via the shared parser) ----


def test_parse_probability_single_tag():
    assert parse_prob("<reasoning>weak</reasoning>\n<probability>0.75</probability>") == 0.75


def test_parse_with_thinking_takes_post_think():
    out = "<think>...0.30...</think>\n<probability>0.30</probability>"
    assert parse_prob(out) == 0.30


def test_clamp_probability():
    assert parse_prob("<probability>1.5</probability>") == 1.0
    assert parse_prob("<probability>0.0</probability>") == 0.0


def test_no_tag_is_none():
    assert parse_prob("I am not sure about this hypothesis.") is None


# ---- rewards (single-target, shared compute_reward / compute_accuracy) ----


def test_brier_reward_single_target():
    assert compute_reward(0.9, 1, reward_type="brier") == pytest.approx(1.0 - (0.9 - 1) ** 2)
    assert compute_reward(0.0, 1, reward_type="brier") == pytest.approx(0.0)


def test_linear_brier_reward_single_target():
    assert compute_reward(0.75, 0, reward_type="linear_brier") == pytest.approx(1.0 - 0.75)


def test_accuracy_single_target():
    assert compute_accuracy(0.9, 1) == 1.0
    assert compute_accuracy(0.1, 1) == 0.0
    assert compute_accuracy(None, 1) == 0.0


def test_build_record_shape():
    row = _episode_row("e1", label=1)
    rec = build_record(row, context_size=4)
    assert rec["task"] == "tahoe_evidence_deliberation"
    assert rec["info"] == {"episode_id": "e1", "label": 1, "context_size": 4}
    assert "Drug DrugX upregulates gene GENE1 in cell line CellY." in rec["question"]


# ---- load_environment: per-split parquet files (train + val loaded, test held out) ----


def _episode_row(eid, label=0):
    return {
        "episode_id": eid,
        "target_drug": "DrugX",
        "target_cell": "CellY",
        "target_gene": "GENE1",
        "target_direction": "upregulates",
        "context_pool": None,
        "label": label,
    }


def _write_splits(tmp_path, by_split):
    for split, rows in by_split.items():
        pd.DataFrame(rows).to_parquet(tmp_path / f"{split}.parquet")
    return str(tmp_path)


def test_loads_train_and_val_and_skips_test(tmp_path):
    path = _write_splits(
        tmp_path,
        {
            "train": [_episode_row("e1"), _episode_row("e2")],
            "val": [_episode_row("e3")],
            "test": [_episode_row("e4")],
        },
    )
    env = load_environment(path, context_size=4, prompt_style="direct")
    assert len(env.dataset) == 2
    assert len(env.eval_dataset) == 1


def test_missing_train_raises(tmp_path):
    path = _write_splits(tmp_path, {"val": [_episode_row("e1")], "test": [_episode_row("e2")]})
    with pytest.raises(FileNotFoundError, match="train"):
        load_environment(path, prompt_style="direct")


def test_missing_val_raises(tmp_path):
    path = _write_splits(tmp_path, {"train": [_episode_row("e1")]})
    with pytest.raises(FileNotFoundError, match="val"):
        load_environment(path, prompt_style="direct")


def test_empty_train_file_raises(tmp_path):
    path = _write_splits(tmp_path, {"train": [], "val": [_episode_row("e1")]})
    with pytest.raises(ValueError, match="train"):
        load_environment(path, prompt_style="direct")
