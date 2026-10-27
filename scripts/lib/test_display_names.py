"""Tests for the shared display-name table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import display_names as names  # noqa: E402


def test_domain_capitalization():
    assert names.domain("tahoe") == "Tahoe"
    assert names.domain("depmap") == "DepMap"
    assert names.domain("gwas") == "GWAS"
    assert names.domain("all") == "All"


def test_metric_names():
    assert names.metric("brier") == "Brier Score"
    assert names.metric("auroc") == "AUROC"
    assert names.metric("smece") == "sMECE"
    assert names.metric("brier_skill_score") == "Brier Skill Score"


def test_unknown_keys_pass_through_rather_than_raising():
    # A missing entry should surface as an odd label on a draft figure, not abort the run.
    assert names.domain("proteomics") == "proteomics"
    assert names.metric("logloss") == "logloss"


def test_model_rows_compose_from_domain_names():
    # Domain capitalization is defined once; row labels must inherit it.
    assert names.model("leave-out-gwas") == "Leave-out GWAS"
    assert names.model("leave-out-depmap") == "Leave-out DepMap"
    assert names.model("leave-out-tahoe") == "Leave-out Tahoe"


def test_all_model_distinguished_from_all_column():
    # The `all` MODEL trained on everything; the `All` COLUMN is a macro-average. Same key, different
    # claims — they must not render identically in a figure that shows both.
    assert names.model("all") == "All domains"
    assert names.domain("all") == "All"
    assert names.model("all") != names.domain("all")


def test_metric_direction():
    assert names.lower_is_better("brier")
    assert names.lower_is_better("ece")
    assert not names.lower_is_better("auroc")
    assert not names.lower_is_better("accuracy")
    # Unknown metrics default to "higher is better" rather than raising.
    assert not names.lower_is_better("mystery")
    assert names.direction_arrow("brier").startswith("↓")
    assert names.direction_arrow("auroc").startswith("↑")


def test_every_metric_key_has_a_known_direction_intent():
    # Guard against adding a loss to METRIC and forgetting LOWER_IS_BETTER: anything whose display name
    # says "Error" or "Score"-as-loss should be reviewed. Here we just assert the sets stay consistent
    # with the documented intent for the metrics the render scripts actually offer.
    for k in ["brier", "ece", "mce", "smece"]:
        assert names.lower_is_better(k), f"{k} is a loss"
    for k in ["accuracy", "auroc", "auprc", "brier_skill_score"]:
        assert not names.lower_is_better(k), f"{k} is a score"
