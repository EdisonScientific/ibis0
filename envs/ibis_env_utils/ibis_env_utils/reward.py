"""Shared reward scoring for ibis evidence-deliberation envs.

`compute_reward` scores a parsed probability against a binary label under a choice of
reward type. Shared so envs over the same evidence-deliberation task score identically.
"""

import math

EPSILON = 1e-7
UNPARSEABLE_REWARD = -1.0


def compute_reward(
    p: float | None,
    label: int,
    *,
    reward_type: str = "likelihood",
    unparseable_reward: float = UNPARSEABLE_REWARD,
    log_lik_clip: float = -10.0,
) -> float:
    """Score a parsed P(supported) ``p`` (or ``None`` if unparseable) against a binary ``label``.

    - ``likelihood`` (default): p_true_class ∈ [0, 1] (improper).
    - ``brier``: 1 - (p - y)^2.
    - ``linear_brier``: 1 - |p - y|.
    - ``log_likelihood_whitened``: strictly proper; log p_true_class clipped at ``log_lik_clip``
      then linearly mapped [log_lik_clip, 0] → [0, 1].
    - ``accuracy_brier``: accuracy (argmax at 0.5) + Brier reward, range [0, 2].
    Unparseable rollouts (``p is None``) return ``unparseable_reward`` (0.0 in the whitened
    case, whose [0, 1] range has no room for a negative penalty).
    """
    if p is None:
        return 0.0 if reward_type == "log_likelihood_whitened" else unparseable_reward
    label = int(label)
    if reward_type == "log_likelihood_whitened":
        pc = max(EPSILON, min(1 - EPSILON, p))
        log_lik = math.log(pc) if label == 1 else math.log(1 - pc)
        clipped = max(log_lik_clip, log_lik)
        return (clipped - log_lik_clip) / (-log_lik_clip)
    if reward_type == "brier":
        return 1.0 - (p - label) ** 2
    if reward_type == "linear_brier":
        return 1.0 - abs(p - label)
    if reward_type == "accuracy_brier":
        accuracy = 1.0 if (p >= 0.5) == (label == 1) else 0.0
        return accuracy + (1.0 - (p - label) ** 2)
    return p if label == 1 else 1 - p


def compute_accuracy(p: float | None, label: int) -> float:
    """Thresholded correctness of a parsed P(supported) ``p`` against a binary ``label``.

    ``1.0`` when the argmax at 0.5 matches the label, else ``0.0``; an unparseable rollout
    (``p is None``) counts as incorrect (``0.0``). Intended as a metrics-only rubric func
    (weight 0) so it's logged without shaping the training reward.
    """
    if p is None:
        return 0.0
    return 1.0 if (p >= 0.5) == (int(label) == 1) else 0.0
