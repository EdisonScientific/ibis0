"""Tests for the canonical eval-output layout."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_paths as ep  # noqa: E402

HUB = "hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}"
REV = "8cc5c6030194b2212e98fa9d5aa5a7b1fda7c3"


def test_hub_testset_slug_pins_the_revision():
    assert ep.testset_slug(HUB, REV) == "EDB-Bio@8cc5c603"


def test_hub_testset_without_revision_is_refused():
    # An unpinned Hub result is indistinguishable from one collected against a later publish.
    with pytest.raises(ValueError, match="explicit revision"):
        ep.testset_slug(HUB)


def test_local_testset_slug_names_the_set_root_not_the_config():
    assert ep.testset_slug("/mnt/u/ibis_data/testset_smoke/primary/{domain}") == "local-testset_smoke"
    assert ep.testset_slug("/mnt/u/ibis_data/testset_smoke/sensitivity/{domain}") == "local-testset_smoke"
    assert ep.testset_slug("/mnt/u/ibis_data/testset_smoke") == "local-testset_smoke"


def test_two_revisions_get_different_slugs():
    a = ep.testset_slug(HUB, "8cc5c603aaaa")
    b = ep.testset_slug(HUB, "ffffffffbbbb")
    assert a != b  # so results cannot land in the same directory


def test_checkpoint_key_carries_run_name_and_step():
    k = ep.checkpoint_key("gwas-depmap-tahoe__qwen3-30b-a3b__rl-direct__r2.1.0", 4200)
    assert k == "gwas-depmap-tahoe__qwen3-30b-a3b__rl-direct__r2.1.0__step_4200"


def test_same_run_different_steps_do_not_collide():
    run = "gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0"
    assert ep.checkpoint_key(run, 2400) != ep.checkpoint_key(run, 3400)


def test_served_key_includes_prompt_style():
    # The base model's two prompt modes are two roster rows; the key must separate them.
    assert ep.served_key("Qwen/Qwen3-30B-A3B", "direct") == "Qwen3-30B-A3B__direct"
    assert ep.served_key("Qwen/Qwen3-30B-A3B", "reasoning") == "Qwen3-30B-A3B__reasoning"
    assert ep.served_key("Qwen/Qwen3-30B-A3B", "direct") != ep.served_key("Qwen/Qwen3-30B-A3B", "reasoning")


def test_served_key_rejects_a_bogus_prompt_style():
    with pytest.raises(ValueError, match="prompt_style"):
        ep.served_key("Qwen/Qwen3-30B-A3B", "thinking")  # the flag value is 'reasoning'


def test_api_key_and_cell_path(tmp_path):
    slug = ep.testset_slug(HUB, REV)
    p = ep.cell_path(slug, ep.api_key("gpt-5.6-sol"), "tahoe", "primary", root=tmp_path)
    assert p == tmp_path / slug / "gpt-5.6-sol" / "tahoe_primary.json"


def test_filenames_carry_no_family_prefix_or_mode(tmp_path):
    slug = ep.testset_slug(HUB, REV)
    key = ep.checkpoint_key("depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0", 1800)
    p = ep.cell_path(slug, key, "gwas", "primary", root=tmp_path)
    assert p.name == "gwas_primary.json"
    assert "classifier" not in p.name and "direct" not in p.name


def test_sensitivity_cells_sit_beside_primary(tmp_path):
    slug, key = ep.testset_slug(HUB, REV), ep.api_key("claude-opus-5")
    a = ep.cell_path(slug, key, "tahoe", "primary", root=tmp_path)
    b = ep.cell_path(slug, key, "tahoe", "ctx7", root=tmp_path)
    assert a.parent == b.parent and a != b


def test_out_prefix_matches_what_the_eval_scripts_append(tmp_path):
    slug, key = ep.testset_slug(HUB, REV), ep.api_key("gpt-5.6-sol")
    pref = ep.out_prefix(slug, key, "depmap", root=tmp_path)
    assert f"{pref}_primary.json" == str(ep.cell_path(slug, key, "depmap", "primary", root=tmp_path))


def test_run_json_refuses_conflicting_provenance(tmp_path):
    slug, key = ep.testset_slug(HUB, REV), ep.checkpoint_key("run__cls__r2.1.0", 2400)
    ep.write_run_json(
        slug, key, {"model_source": "/models/a", "prompt_style": "direct", "checkpoint_step": 2400}, root=tmp_path
    )
    # Same directory, different model: this is how a figure silently mixes two models.
    with pytest.raises(ValueError, match="different provenance"):
        ep.write_run_json(
            slug, key, {"model_source": "/models/b", "prompt_style": "direct", "checkpoint_step": 2400}, root=tmp_path
        )


def test_run_json_merges_cell_lists_for_a_matching_rerun(tmp_path):
    slug, key = ep.testset_slug(HUB, REV), ep.api_key("gpt-5.6-sol")
    base = {"model_source": "gpt-5.6-sol", "prompt_style": "reasoning", "checkpoint_step": None}
    ep.write_run_json(slug, key, {**base, "cells_written": ["tahoe_primary"]}, root=tmp_path)
    p = ep.write_run_json(slug, key, {**base, "cells_written": ["gwas_primary"]}, root=tmp_path)
    assert json.loads(p.read_text())["cells_written"] == ["gwas_primary", "tahoe_primary"]


def test_run_json_accumulates_testset_configs_and_drops_the_stale_scalar(tmp_path):
    # primary and sensitivity share a slug by design, so one directory spans both configs. run.json
    # must not end up naming only the config that happened to run last.
    slug, key = ep.testset_slug(HUB, REV), ep.api_key("gpt-5.6-sol")
    base = {"model_source": "gpt-5.6-sol", "prompt_style": "reasoning", "checkpoint_step": None}
    sens = HUB.replace("/primary/", "/sensitivity/")
    p = ep.write_run_json(slug, key, {**base, "testset": HUB}, root=tmp_path)
    assert json.loads(p.read_text())["testset"] == HUB  # one config -> scalar is the whole truth
    p = ep.write_run_json(slug, key, {**base, "testset": sens}, root=tmp_path)
    rec = json.loads(p.read_text())
    assert rec["testsets"] == sorted([HUB, sens])
    assert "testset" not in rec  # dropped rather than left describing half the directory


def test_run_json_single_config_rerun_keeps_the_scalar(tmp_path):
    slug, key = ep.testset_slug(HUB, REV), ep.api_key("claude-opus-5")
    base = {"model_source": "claude-opus-5", "prompt_style": "reasoning", "testset": HUB}
    ep.write_run_json(slug, key, base, root=tmp_path)
    rec = json.loads(ep.write_run_json(slug, key, base, root=tmp_path).read_text())
    assert rec["testset"] == HUB and rec["testsets"] == [HUB]


def test_iter_model_keys_is_the_reader_side(tmp_path):
    slug = ep.testset_slug(HUB, REV)
    for key in ["Qwen3-30B-A3B__direct", "gpt-5.6-sol"]:
        ep.cell_path(slug, key, "tahoe", "primary", root=tmp_path).parent.mkdir(parents=True, exist_ok=True)
    assert ep.iter_model_keys(slug, root=tmp_path) == ["Qwen3-30B-A3B__direct", "gpt-5.6-sol"]
    assert ep.iter_model_keys("nonexistent", root=tmp_path) == []
