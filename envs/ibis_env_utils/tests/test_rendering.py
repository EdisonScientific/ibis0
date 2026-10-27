"""Unit tests for the shared data->prompt rendering pipeline (ibis_env_utils.rendering)."""

import verifiers as vf
from ibis_env_utils import (
    DEFAULT_FOOTER,
    build_single_target_record,
    make_evidence_rubric,
    parse_prob,
    render_evidence_prompt,
)


def _stub_renderer(idx, stub):
    return [f"\n[{idx}] {stub['rel']}", f"  value: {stub['v']}"]


def test_render_evidence_prompt_layout():
    stubs = [{"rel": "same", "v": 1}, {"rel": "control", "v": 2}]
    out = render_evidence_prompt(stubs, stub_renderer=_stub_renderer, target_block="TARGET LINE")
    assert out.startswith("CONTEXT EVIDENCE\n================")
    assert "\n[1] same\n  value: 1" in out
    assert "\n[2] control\n  value: 2" in out
    assert "TARGET HYPOTHESIS\n=================" in out
    assert "TARGET LINE" in out
    assert out.endswith(DEFAULT_FOOTER)
    # header, target header, and footer are separated by a blank line
    assert "=================\nTARGET LINE\n\n" + DEFAULT_FOOTER in out


def test_render_evidence_prompt_empty_notice():
    out = render_evidence_prompt([], stub_renderer=_stub_renderer, target_block="T", empty_context_notice="(none)")
    assert "(none)" in out
    # no numbered stubs
    assert "[1]" not in out


def test_render_evidence_prompt_empty_no_notice():
    out = render_evidence_prompt([], stub_renderer=_stub_renderer, target_block="T")
    assert "(none)" not in out
    assert "CONTEXT EVIDENCE\n================\n\nTARGET HYPOTHESIS" in out


def test_render_evidence_prompt_custom_footer():
    out = render_evidence_prompt([], stub_renderer=_stub_renderer, target_block="T", footer="FOOT")
    assert out.endswith("\nFOOT")


def _render_user(row, *, prompt_style="reasoning"):
    return f"USER n={len(row['context_pool'])} style={prompt_style}"


def test_build_single_target_record_shape_and_slicing():
    row = {"episode_id": "e1", "label": 1, "context_pool": [0, 1, 2, 3, 4], "ood_group": "gene"}
    rec = build_single_target_record(
        row, 3, task="x_evidence_deliberation", render_user=_render_user, prompt_style="direct"
    )
    assert rec["task"] == "x_evidence_deliberation"
    assert rec["answer"] == ""
    assert rec["question"] == "USER n=3 style=direct"  # pool sliced to context_size, style threaded
    assert rec["info"] == {"episode_id": "e1", "label": 1, "context_size": 3}


def test_build_single_target_record_extra_info_and_none_pool():
    row = {"episode_id": "e2", "label": 0, "context_pool": None, "ood_group": "cell"}
    rec = build_single_target_record(
        row,
        4,
        task="t",
        render_user=_render_user,
        extra_info=lambda r: {"ood_group": r.get("ood_group")},
    )
    assert rec["question"] == "USER n=0 style=reasoning"  # None pool -> empty, default style
    assert rec["info"] == {"episode_id": "e2", "label": 0, "context_size": 4, "ood_group": "cell"}


def test_parse_prob_basic_and_clamp_and_think():
    assert parse_prob("<probability>0.73</probability>") == 0.73
    assert parse_prob("<probability>1.5</probability>") == 1.0
    assert parse_prob("no tag here") is None
    # last tag wins, and only the post-</think> text is parsed
    assert parse_prob("<probability>0.1</probability></think><probability>0.9</probability>") == 0.9
    assert parse_prob("a <probability>0.2</probability> b <probability>0.8</probability>") == 0.8


def test_make_evidence_rubric_brier_parity():
    parser = vf.Parser()
    rubric = make_evidence_rubric(reward_type="brier", unparseable_reward=-1.0, log_lik_clip=-10.0, parser=parser)
    funcs = {f.__name__: f for f in rubric.funcs}
    # brier reward reproduces the old single-target 1 - (p - y)^2
    for p, y in [(0.9, 1), (0.2, 0), (0.5, 1), (0.75, 0)]:
        comp = f"<probability>{p}</probability>"
        assert abs(funcs["evidence_reward"](completion=comp, info={"label": y}, state={}) - (1.0 - (p - y) ** 2)) < 1e-9
    # format + accuracy
    assert funcs["format_reward"](completion="<probability>0.5</probability>", info={"label": 1}) == 1.0
    assert funcs["format_reward"](completion="no tag", info={"label": 1}) == 0.0
    assert funcs["accuracy"](completion="<probability>0.9</probability>", info={"label": 1}) == 1.0
    assert funcs["accuracy"](completion="<probability>0.1</probability>", info={"label": 1}) == 0.0
    # weights and func ordering match the shared contract
    assert [f.__name__ for f in rubric.funcs] == ["evidence_reward", "format_reward", "accuracy"]
    assert rubric.weights == [0.9, 0.1, 0.0]


def test_make_evidence_rubric_unparseable():
    parser = vf.Parser()
    rubric = make_evidence_rubric(reward_type="brier", unparseable_reward=-1.0, log_lik_clip=-10.0, parser=parser)
    funcs = {f.__name__: f for f in rubric.funcs}
    assert funcs["evidence_reward"](completion="no probability", info={"label": 1}, state={}) == -1.0
