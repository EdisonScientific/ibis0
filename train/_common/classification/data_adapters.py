"""Data layer for the shared classification-head trainer.

One trainer, three data sources. Each source (GWAS, DepMap, Tahoe) renders its RL episodes into the
same `{episode_id, question, label, system_prompt}` records; the differences are only in *how* those
records are produced. A `DataAdapter` encapsulates one source's render path, so the trainer stays
domain-agnostic and reads the same episodes/prompts as the corresponding RL env.

All three domains use a single adapter, `IbisEnvAdapter`, parameterized by env-module name: since
gwas / depmap / tahoe share the `ibis_env_utils` load + `build_record` + `system_prompt_for`
contract, one render path covers them. It loads splits with `ibis_env_utils.load_dataset_split` and
renders each row via the env's `build_record`, tagging every record with the env's
`system_prompt_for(prompt_style)`.

`prompt_style` (direct vs. reasoning) applies to all three domains — each env module exposes a
`system_prompt_for`.

Each adapter also carries `model_forward_kwargs` (extra kwargs threaded into the model forward);
today it is empty for every adapter — the shared Qwen3-MoE `forward()` takes no extra kwargs — but
the per-record hook is kept for future model-specific needs. `encode_record` and the infinite sampler
are identical across domains and live here as shared helpers.
"""

import importlib
import math
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor

if TYPE_CHECKING:
    from train_classification_head import ClassificationConfig


class ClassificationDataset:
    """Infinite per-rank sampler over training records, reshuffled each epoch."""

    def __init__(self, records: list[dict], seed: int, data_rank: int, data_world_size: int):
        self.records = records
        self.seed = seed
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.epoch = 0
        self.position = 0
        self.indices = self._indices_for_epoch()

    def _indices_for_epoch(self) -> list[int]:
        indices = list(range(len(self.records)))
        random.Random(self.seed + self.epoch).shuffle(indices)
        return indices[self.data_rank :: self.data_world_size]

    def next(self) -> dict:
        if self.position >= len(self.indices):
            self.epoch += 1
            self.position = 0
            self.indices = self._indices_for_epoch()
        idx = self.indices[self.position]
        self.position += 1
        return self.records[idx]


class MixtureDataset:
    """Weighted mixture over per-domain infinite samplers (equal weight by default).

    Mirrors the RL orchestrator's `TrainSource` (orchestrator/train_source.py): holds one per-rank
    `ClassificationDataset` per domain and, on `next()`, picks a domain via a seeded weighted choice
    then pulls that domain's next record. `weights` maps domain -> relative weight (normalized
    internally); `None` means uniform (1:1:...). The domain-choice RNG is offset by `data_rank` so
    data-parallel ranks don't lock-step onto the same domain each microstep — every optimizer step
    then sees a domain mix across the DP + grad-accum dimensions rather than a single domain.
    """

    def __init__(
        self,
        records_by_domain: dict[str, list[dict]],
        weights: dict[str, float] | None,
        seed: int,
        data_rank: int,
        data_world_size: int,
    ):
        self.domains = list(records_by_domain.keys())
        self.samplers = {
            domain: ClassificationDataset(records, seed, data_rank, data_world_size)
            for domain, records in records_by_domain.items()
        }
        self.weights = [1.0 if weights is None else weights[domain] for domain in self.domains]
        self.rng = random.Random(seed + 1000 * (data_rank + 1))

    def next(self) -> dict:
        domain = self.rng.choices(self.domains, weights=self.weights, k=1)[0]
        return self.samplers[domain].next()


class DataAdapter(Protocol):
    """One classification data source's render path.

    `display_name` is used in log lines only. `model_forward_kwargs` is threaded into the model
    forward; empty for all current adapters (the shared Qwen3-MoE forward takes no extra kwargs).
    `render_split` yields the `{episode_id, question, label, system_prompt}` records for one split.
    """

    display_name: str
    model_forward_kwargs: dict

    def render_split(self, config: "ClassificationConfig", data_path: str, split: str) -> list[dict]: ...


class IbisEnvAdapter:
    """Adapter for env modules that expose the ibis_env_utils load + `build_record` render path.

    Parameterized by env-module name (`gwas_meta` / `depmap_meta` / `tahoe_meta`). Renders via
    `load_dataset_split(data_path, split)` -> `build_record(row, context_size)` ->
    `{episode_id, question, label, system_prompt}` with the env's `system_prompt_for(config.prompt_style)`
    (default `"direct"`).
    """

    model_forward_kwargs: dict = {}

    def __init__(self, env_module_name: str, display_name: str):
        self.env_module_name = env_module_name
        self.display_name = display_name

    def render_split(self, config: "ClassificationConfig", data_path: str, split: str) -> list[dict]:
        from ibis_env_utils import Split, context_size_for, load_dataset_split

        env = importlib.import_module(self.env_module_name)
        split_enum = {"train": Split.TRAIN, "val": Split.VAL}[split]
        df = load_dataset_split(data_path, split_enum)
        # Per-episode context length (fixed, or a seeded draw over config.context_size_range) — matches
        # the RL env's render for the same episode_id.
        records = [
            env.build_record(
                row,
                context_size_for(
                    row.get("episode_id"), config.context_size, config.context_size_range, config.context_seed
                ),
            )
            for row in df.to_dict(orient="records")
        ]
        system_prompt = env.system_prompt_for(config.prompt_style)
        return [
            {
                "episode_id": r["info"]["episode_id"],
                "question": r["question"],
                "label": r["info"]["label"],
                "system_prompt": system_prompt,
            }
            for r in records
        ]


# domain selector -> adapter. All three domains share IbisEnvAdapter (parameterized by env-module
# name): every env exposes the same `build_record` + `system_prompt_for` contract.
def get_adapter(domain: str) -> DataAdapter:
    """Resolve the data adapter for a domain selector (`gwas` / `depmap` / `tahoe`)."""
    adapters: dict[str, DataAdapter] = {
        "gwas": IbisEnvAdapter("gwas_meta", "GWAS"),
        "depmap": IbisEnvAdapter("depmap_meta", "DepMap"),
        "tahoe": IbisEnvAdapter("tahoe_meta", "Tahoe"),
    }
    return adapters[domain]


def _render_domain_split(config: "ClassificationConfig", spec, split: str) -> list[dict]:
    """Render one domain's split into records, tagged with `domain` + `forward_kwargs`.

    Resolves the adapter, renders the split (train on `train`, select on `val`; `test` is never
    loaded — held out for final reporting), applies the split's `max_*_examples` truncation, and tags
    every record with its `domain` and the adapter's `forward_kwargs` (empty for all current
    adapters) so a mixed batch carries the right per-record forward kwargs.
    """
    adapter = get_adapter(spec.domain)
    records = adapter.render_split(config, spec.data_path, split)
    if not records:
        raise ValueError(f"no rows in the '{split}' split of {spec.data_path}")
    limit = config.max_train_examples if split == "train" else config.max_eval_examples
    if limit is not None:
        records = records[:limit]
    forward_kwargs = adapter.model_forward_kwargs
    for record in records:
        record["domain"] = spec.domain
        record["forward_kwargs"] = forward_kwargs
    return records


def build_all_records(
    config: "ClassificationConfig",
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Render train records for `config.domains` and val records for `domains` + `held_out`.

    Training pools come from the trained `domains`; val pools cover the union of the trained domains
    and the held-out ones (`config.held_out`), so a held-out domain — never trained on — still gets
    its `val` split loaded and evaluated (the cross-domain transfer measurement). The two lists are
    disjoint (validated), so keys don't collide; `val_by_domain` is ordered trained-first, then
    held-out. Returns `(train_by_domain, val_by_domain)`, each keyed by domain.
    """
    train_by_domain = {spec.domain: _render_domain_split(config, spec, "train") for spec in config.domains}
    val_specs = [*config.domains, *config.held_out]
    val_by_domain = {spec.domain: _render_domain_split(config, spec, "val") for spec in val_specs}
    return train_by_domain, val_by_domain


def encode_record(
    record: dict, tokenizer, max_seq_len: int, pad_to_multiple: int
) -> tuple[Tensor, Tensor, int, Tensor]:
    """Tokenize one record to padded (input_ids, position_ids, last-token index, label)."""
    messages = [
        {"role": "system", "content": record["system_prompt"]},
        {"role": "user", "content": record["question"]},
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(token_ids, Mapping):  # dict or HF BatchEncoding (a UserDict, not a dict)
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if len(token_ids) > max_seq_len:
        raise ValueError(
            f"Episode {record['episode_id']} renders to {len(token_ids)} tokens, exceeding "
            f"max_seq_len={max_seq_len}. The classification head reads the final token's logit, so "
            "tail-truncating to fit would silently drop the leading system prompt (the task framing) and "
            "break the input/label correspondence. Raise model.seq_len or lower data.context_size instead."
        )
    real_len = len(token_ids)
    padded_len = math.ceil(real_len / pad_to_multiple) * pad_to_multiple
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    token_ids = token_ids + [pad_id] * (padded_len - real_len)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    position_ids = torch.arange(padded_len, dtype=torch.long, device="cuda").unsqueeze(0)
    label = torch.tensor([float(record["label"])], dtype=torch.float32, device="cuda")
    return input_ids, position_ids, real_len - 1, label
