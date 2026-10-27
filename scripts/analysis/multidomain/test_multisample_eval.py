"""The multi-draw reduction in eval_generative_testset.

Two behaviours here decide every number in the benchmark, and both fail silently if wrong: the reported
probability is the mean of the draws that parsed, and an episode counts as unreadable only when every
draw failed. The second is the one worth pinning -- discarding an episode because one draw of four came
back malformed would bias the retained set toward prompts the model finds easy to format, which is the
same survivorship problem the parse-failure convention exists to avoid.

The reduction is inlined in the eval's main loop rather than factored out, so these tests re-express it
and check the properties. If the loop changes, this file must change with it.
"""

import numpy as np
import pytest


def reduce_draws(draws: list[float | None]) -> tuple[float | None, int, float | None]:
    """(mean of parsed draws, how many parsed, their sd) -- mirrors the eval's per-episode reduction."""
    got = [d for d in draws if d is not None]
    return (
        float(np.mean(got)) if got else None,
        len(got),
        float(np.std(got, ddof=1)) if len(got) > 1 else None,
    )


def test_mean_of_all_parsed():
    p, n, sd = reduce_draws([0.2, 0.4, 0.6, 0.8])
    assert p == pytest.approx(0.5)
    assert n == 4
    assert sd == pytest.approx(np.std([0.2, 0.4, 0.6, 0.8], ddof=1))


def test_partial_failure_averages_the_survivors():
    """One bad draw is a decoding hiccup, not a refusal: the episode still scores."""
    p, n, _ = reduce_draws([0.2, None, 0.6, None])
    assert p == pytest.approx(0.4)
    assert n == 2


def test_episode_fails_only_when_every_draw_fails():
    p, n, sd = reduce_draws([None, None, None, None])
    assert p is None
    assert n == 0
    assert sd is None


def test_single_surviving_draw_has_no_sd():
    """sd is undefined for one observation; it must be None rather than 0.0, which would read as agreement."""
    p, n, sd = reduce_draws([None, 0.7, None, None])
    assert p == pytest.approx(0.7)
    assert n == 1
    assert sd is None


def test_mean_is_taken_over_probabilities_not_logits():
    """Averaging in probability space is the convention the metrics assume; averaging in logit space
    would move the reported value and silently change every calibration number."""
    p, _, _ = reduce_draws([0.01, 0.99])
    assert p == pytest.approx(0.5)


def test_reduction_is_order_invariant():
    a, _, _ = reduce_draws([0.1, None, 0.9])
    b, _, _ = reduce_draws([0.9, 0.1, None])
    assert a == pytest.approx(b)
