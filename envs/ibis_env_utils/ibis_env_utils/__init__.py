"""Shared helpers for ibis verifiers envs.

Surfaces:

- `load_dataset_split` / `Split`: load one split (`train`/`val`/`test`) of a collection
  from an hf:// repo or local dir of per-split parquet files; the split is which frame
  you ask for, not a column to filter (see `dataset.py`).
- `split_train_eval`: deterministic positional train/eval split (see `split.py`).
- `context_size_for` / `validate_context_size_range`: per-episode context length — fixed, or a
  deterministic seeded uniform draw over a range keyed by `episode_id` (see `context.py`).
- `state_text_length`: generic rubric-func factory (see `metrics.py`).
- `compute_reward`: shared probability-vs-label reward scoring (see `reward.py`).
- `build_probability_system_prompts` / `select_system_prompt` / `DIRECT_INSTRUCTION` /
  `REASONING_INSTRUCTION`: shared direct/reasoning answer-format prompting (see `prompts.py`).
"""

from ibis_env_utils.context import context_size_for, validate_context_size_range
from ibis_env_utils.dataset import Split, load_dataset_split
from ibis_env_utils.metrics import state_text_length
from ibis_env_utils.prompts import (
    DIRECT_INSTRUCTION,
    PRIOR_NOTE,
    REASONING_INSTRUCTION,
    build_probability_system_prompts,
    select_system_prompt,
)
from ibis_env_utils.rendering import (
    DEFAULT_FOOTER,
    build_single_target_record,
    make_evidence_rubric,
    parse_prob,
    render_evidence_prompt,
)
from ibis_env_utils.reward import UNPARSEABLE_REWARD, compute_accuracy, compute_reward
from ibis_env_utils.split import split_train_eval

__all__ = [
    "DEFAULT_FOOTER",
    "DIRECT_INSTRUCTION",
    "PRIOR_NOTE",
    "REASONING_INSTRUCTION",
    "Split",
    "UNPARSEABLE_REWARD",
    "build_probability_system_prompts",
    "build_single_target_record",
    "compute_accuracy",
    "compute_reward",
    "context_size_for",
    "load_dataset_split",
    "make_evidence_rubric",
    "parse_prob",
    "render_evidence_prompt",
    "validate_context_size_range",
    "select_system_prompt",
    "split_train_eval",
    "state_text_length",
]
