"""Tests for the sweep runner in eval_frontier_testset.

`_run_passes` has two properties that are invisible when broken:

- it must issue calls **k-major** (all episodes at one context size before moving on); an episode-major
  variant was tried and reverted because it made provider caching worse.
- it must journal every completed call, so a crash resumes instead of re-paying for thousands of requests.

`_chat` is stubbed out, so these run with no API client and no network.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_frontier_testset as F  # noqa: E402


def _args(**over):
    base = dict(
        model="claude-opus-5",
        provider="anthropic",
        data_path="hf://org/repo/sensitivity/tahoe",
        reasoning_effort="high",
        max_output_tokens=24000,
        prompt_style="reasoning",
        limit=None,
        domain="tahoe",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _passes(n_ep=3, k_max=2):
    """`[(tag, rows)]` ascending in k, mirroring what testset_passes yields for a dense config."""
    return [
        (
            f"ctx{k}",
            [{"episode_id": f"ep{i}", "label": i % 2, "n_stubs": k, "ground_truth": None} for i in range(n_ep)],
        )
        for k in range(k_max + 1)
    ]


@pytest.fixture
def spy(monkeypatch):
    """Replace _chat with a recorder; returns the list of (episode_id, tag) in call order."""
    calls = []

    async def fake_chat(client, args, user, sem, sysprompt):
        calls.append(user)  # render_user encodes "<eid>@<tag>"
        return "<probability>0.5</probability>", "ok", {"input_tokens": 10, "output_tokens": 2}

    monkeypatch.setattr(F, "_chat", fake_chat)
    return calls


def _run(args, passes, out_dir, conc=8):
    return asyncio.run(
        F._run_passes(
            client=None,
            args=args,
            passes=passes,
            sem=asyncio.Semaphore(conc),
            sysprompt="sys",
            render_user=lambda r: f"{r['episode_id']}@{r['n_stubs']}",
            parse_prob=lambda t: 0.5,
            out_dir=out_dir,
        )
    )


def test_calls_are_issued_k_major(spy, tmp_path):
    """Every episode at one context size before any move to the next.

    Episode-major ordering would interleave (ep0@0, ep0@1, ...); k-major groups by k.
    """
    per_tag = _run(_args(), _passes(n_ep=4, k_max=4), tmp_path)
    ks = [int(u.split("@")[1]) for u in spy]
    assert ks == sorted(ks), f"context sizes not issued in k-major blocks: {ks}"
    assert ks == [k for k in range(5) for _ in range(4)]
    assert set(per_tag) == {"ctx0", "ctx1", "ctx2", "ctx3", "ctx4"}


def test_every_cell_gets_every_episode(spy, tmp_path):
    """Regrouping must not drop or duplicate work: 3 episodes x 3 k = 9 calls, 3 per cell."""
    per_tag = _run(_args(), _passes(n_ep=3, k_max=2), tmp_path)
    assert len(spy) == 9
    for tag in ("ctx0", "ctx1", "ctx2"):
        assert sorted(e[0]["episode_id"] for e in per_tag[tag]) == ["ep0", "ep1", "ep2"]


def test_results_are_ordered_deterministically_within_a_cell(spy, tmp_path):
    """Chains finish in arbitrary order, so cells are sorted by episode_id before writing."""
    per_tag = _run(_args(), _passes(n_ep=6, k_max=1), tmp_path)
    for tag, entries in per_tag.items():
        ids = [e[0]["episode_id"] for e in entries]
        assert ids == sorted(ids), tag


def test_journal_records_a_header_and_every_call(spy, tmp_path):
    _run(_args(), _passes(n_ep=3, k_max=2), tmp_path)
    lines = (tmp_path / "tahoe_journal.jsonl").read_text().splitlines()
    head = json.loads(lines[0])
    assert head["_header"] and head["model"] == "claude-opus-5"
    assert len(lines) == 1 + 9
    recs = [json.loads(ln) for ln in lines[1:]]
    assert {(r["tag"], r["episode_id"]) for r in recs} == {(f"ctx{k}", f"ep{i}") for k in range(3) for i in range(3)}


def test_a_matching_journal_resumes_without_recalling(spy, tmp_path):
    """The point of the journal: a second run of the same config issues zero new API calls."""
    args, passes = _args(), _passes(n_ep=3, k_max=2)
    _run(args, passes, tmp_path)
    assert len(spy) == 9
    spy.clear()
    per_tag = _run(args, passes, tmp_path)
    assert spy == [], "resume re-issued calls that were already journalled"
    assert sum(len(v) for v in per_tag.values()) == 9


def test_a_journal_from_a_different_run_config_is_ignored(spy, tmp_path):
    """A journal keyed to another model/effort must not be adopted — that would mix rosters in one cell."""
    passes = _passes(n_ep=2, k_max=1)
    _run(_args(model="claude-opus-5"), passes, tmp_path)
    assert len(spy) == 4
    spy.clear()
    _run(_args(model="gpt-5.6-sol"), passes, tmp_path)  # same paths, different model
    assert len(spy) == 4, "records from a different run config were silently reused"


def test_a_torn_final_journal_line_only_costs_that_call(spy, tmp_path):
    """A kill mid-write leaves a partial line; it must be skipped, not crash the resume."""
    args, passes = _args(), _passes(n_ep=2, k_max=1)
    _run(args, passes, tmp_path)
    jp = tmp_path / "tahoe_journal.jsonl"
    jp.write_text(jp.read_text() + '{"tag": "ctx1", "epi')  # truncated
    spy.clear()
    _run(args, passes, tmp_path)
    assert spy == []


def test_limit_truncates_episodes_not_k(spy, tmp_path):
    """--limit is a cost pilot: fewer episodes, each still swept over the full k grid."""
    per_tag = _run(_args(limit=2), _passes(n_ep=5, k_max=2), tmp_path)
    assert len(spy) == 6  # 2 episodes x 3 k
    for tag in per_tag:
        assert len(per_tag[tag]) == 2
