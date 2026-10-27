"""Per-episode context-length sampling, shared across the ibis envs and the classifier.

Evidence-deliberation episodes render a prompt from the first K stubs of a fixed, pre-shuffled
`context_pool` (`pool[:K]`). K is either a single fixed `context_size` or, to train (and evaluate) on
a *distribution* of context lengths, drawn per episode by `context_size_for` from a range — deterministically keyed by `episode_id`, so a given episode gets the same K across
runs, models, and epochs (reproducible, and identical between the RL env and the classifier reading
the same episode). Shared so the training and evaluation render paths draw identical K.
"""

import hashlib
from collections.abc import Sequence


def validate_context_size_range(context_size_range: Sequence[int] | None) -> tuple[int, int] | None:
    """Validate a `(lo, hi)` context range: exactly two ints with `0 <= lo <= hi`. `None` passes."""
    if context_size_range is None:
        return None
    if len(context_size_range) != 2:
        raise ValueError(f"context_size_range must be (lo, hi), got {context_size_range!r}")
    lo, hi = int(context_size_range[0]), int(context_size_range[1])
    if not (0 <= lo <= hi):
        raise ValueError(f"context_size_range must satisfy 0 <= lo <= hi, got ({lo}, {hi})")
    return lo, hi


def context_size_for(
    episode_id: object,
    context_size: int = 16,
    context_size_range: Sequence[int] | None = None,
    seed: int = 0,
) -> int:
    """Context length K for one episode.

    Returns the fixed `context_size` when `context_size_range` is `None`; otherwise a deterministic
    uniform draw in the inclusive range `{lo..hi}`, keyed by `episode_id` (and `seed`) so the same
    episode always maps to the same K. Callers slice `context_pool[:K]`.
    """
    if context_size_range is None:
        return context_size
    lo, hi = context_size_range
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()
    return lo + int(digest, 16) % (hi - lo + 1)
