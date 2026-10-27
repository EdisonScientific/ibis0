import pytest
from ibis_env_utils import (
    DIRECT_INSTRUCTION,
    REASONING_INSTRUCTION,
    build_probability_system_prompts,
    select_system_prompt,
    split_train_eval,
    state_text_length,
)


def test_build_probability_system_prompts_appends_shared_endings():
    preamble = "PREAMBLE.\n"
    reasoning, direct = build_probability_system_prompts(preamble)
    assert reasoning == preamble + REASONING_INSTRUCTION
    assert direct == preamble + DIRECT_INSTRUCTION
    # both keep the answer tag; only direct forbids prose
    assert reasoning.endswith("<probability>0.XX</probability>")
    assert direct.endswith("<probability>0.XX</probability>")
    assert "no reasoning, no explanation" in direct
    assert "no reasoning, no explanation" not in reasoning


def test_select_system_prompt_maps_style_and_rejects_unknown():
    assert select_system_prompt("direct", reasoning="R", direct="D") == "D"
    assert select_system_prompt("reasoning", reasoning="R", direct="D") == "R"
    with pytest.raises(ValueError, match="prompt_style"):
        select_system_prompt("bogus", reasoning="R", direct="D")


def test_split_train_eval_deterministic():
    records = list(range(20))
    train_a, eval_a = split_train_eval(records, n_eval=5, seed=0)
    train_b, eval_b = split_train_eval(records, n_eval=5, seed=0)
    assert (train_a, eval_a) == (train_b, eval_b)
    assert len(eval_a) == 5
    assert len(train_a) == 15
    # Partition is a disjoint cover of the input.
    assert set(train_a) | set(eval_a) == set(records)
    assert not (set(train_a) & set(eval_a))
    # Train preserves original order.
    assert train_a == sorted(train_a)


def test_split_train_eval_seed_changes_holdout():
    records = list(range(50))
    _, eval0 = split_train_eval(records, 10, seed=0)
    _, eval1 = split_train_eval(records, 10, seed=1)
    assert set(eval0) != set(eval1)


def test_split_train_eval_degenerate_n_eval():
    records = list(range(5))
    assert split_train_eval(records, n_eval=0) == (records, [])
    assert split_train_eval(records, n_eval=10) == (records, [])


def test_state_text_length():
    fn = state_text_length("judge_text")
    assert fn.__name__ == "length_judge_text"
    assert fn(state=None) == 0.0
    assert fn(state={"judge_text": "hello"}) == 5.0
    assert fn(state={}) == 0.0
