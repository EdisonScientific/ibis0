"""Deterministic train/eval split for the eval-split convention.

Every env's `load_environment(...)` accepts `n_eval: int = 0` and
`eval_seed: int = 0`. When `n_eval > 0`, the env deterministically holds out
`n_eval` records and returns a `vf.SingleTurnEnv` with both `dataset=` (train)
and `eval_dataset=` (held-out) populated. prime-rl already routes
`[[orchestrator.train.env]]` → `env.get_dataset()` (train side) and
`[[orchestrator.eval.env]]` → `env.get_eval_dataset(n=...)` (eval side), so no
orchestrator-side changes are required.
"""

import random
from typing import TypeVar

T = TypeVar("T")


def split_train_eval(
    records: list[T],
    n_eval: int,
    seed: int = 0,
) -> tuple[list[T], list[T]]:
    """Deterministically split records into (train, eval) lists.

    The eval slice is the first `n_eval` records of a seeded shuffle of the
    indices of `records`. Train is everything else, in original order. Both
    sides of a train/eval pair (i.e. the `[[orchestrator.train.env]]` and
    `[[orchestrator.eval.env]]` instances of the same env) must be loaded with
    the same `n_eval` and `seed` values to produce identical partitions.

    Args:
        records: list of any per-example dicts (or other records) the env
            wants to split.
        n_eval: number of records to hold out for eval. If 0 or larger than
            len(records), `eval` will be empty and `train` returns all records.
        seed: random seed for the shuffle. Default 0.

    Returns:
        (train, eval): two lists. `train` preserves the original order of the
        records that didn't land in the eval slice. `eval` is in shuffled
        order (it's a held-out set; no caller should depend on its order).
    """
    if n_eval <= 0 or n_eval >= len(records):
        return list(records), []

    rng = random.Random(seed)
    shuffled_indices = list(range(len(records)))
    rng.shuffle(shuffled_indices)

    held_out_idx = set(shuffled_indices[:n_eval])
    train = [r for i, r in enumerate(records) if i not in held_out_idx]
    held_out = [records[i] for i in shuffled_indices[:n_eval]]
    return train, held_out
