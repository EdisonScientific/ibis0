"""Supervised BCE classification-head trainer for the evidence-deliberation tasks.

Swaps a large model's LM head for a scalar head and trains it (last-token logit, BCE loss) on the
same examples/prompts as the RL env. One trainer serves all domains (GWAS, DepMap, Tahoe) and any
mixture of them: the `[[domains]]` config list selects one or more data adapters (see
data_adapters.py) that render each domain's episodes into records, which a weighted `MixtureDataset`
interleaves; eval runs per-domain (`eval/<domain>/...`), so training on a subset while evaluating on
all domains — the cross-domain transfer setup — falls out directly. The training loop, model, and
distributed plumbing are shared. The train()/main() structure follows the prime-rl
trainer/entrypoint pattern.

Invoke with one or more domains in the config (single-domain scalar `domain`/`data_path` still works):
    torchrun ... train/_common/classification/train_classification_head.py @ <config.toml>
"""

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal

import prime_rl._compat  # noqa: F401
import torch
import torch.distributed as dist
import torch.nn.functional as F
from best_ckpt import ClassificationCheckpointConfig, get_metrics_dir, load_metric_history, should_early_stop
from data_adapters import MixtureDataset, build_all_records, encode_record
from model_utils import (
    ScalarClassificationHead,
    aggregate,
    classifier_logits,
    save_checkpoint,
    split_named_parameters_by_type,
    split_parameters_by_grad_type,
    write_metric_snapshot,
)
from prime_rl.configs.shared import TrainerLogConfig, WandbConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
)
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.model import DTYPE_MAP, setup_model, setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import ParallelDims, get_parallel_dims
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit
from pydantic import Field, model_validator
from torchtitan.distributed.utils import clip_grad_norm_


# One training domain: which adapter + where its data lives. Parallels the RL side's EnvConfig
# ([[orchestrator.train.env]] with `ratio`); `weight` is this domain's relative mixture weight.
class DomainSpec(BaseConfig):
    """A single training domain for the classifier.

    `domain` selects the data adapter (`gwas` / `depmap` / `tahoe`, all via ibis_env_utils + the env's
    `build_record`). `data_path` is that domain's `hf://<org>/<repo>/<config>` URI
    (or a local dir of per-split parquet files). `weight` is the relative mixture weight — set on all
    domains or none; unset everywhere means equal (1:1:...) sampling.
    """

    domain: Literal["gwas", "depmap", "tahoe"]
    data_path: str
    weight: float | None = Field(None, gt=0)


# Assembled from prime-rl config types (ModelConfig, OptimizerConfig, ...). Mirrors the SFTConfig layout in prime_rl/configs/sft.py.
class ClassificationConfig(BaseConfig):
    """Top-level config for a classification-head training run.

    `domains` (a `[[domains]]` list) is the **training mix** — one or more `DomainSpec`s, each with
    its own `domain` selector + `data_path`, mirroring the RL side's `[[orchestrator.train.env]]`.
    Multiple domains are mixed by `MixtureDataset` (weighted by each spec's `weight`, equal by
    default); a single domain gives a single-domain run. For backward compatibility the
    scalar `domain` + `data_path` fields are still accepted and are normalized into a one-element
    `domains` list.

    `held_out` (an optional `[[held_out]]` list) names domains to **evaluate but not train on** — the
    cross-domain-transfer setup: e.g. train on gwas+depmap, hold out tahoe. Held-out domains must be
    disjoint from the trained `domains` and carry no `weight` (they never enter the training mix).
    Eval always covers the union `domains ∪ held_out`, so both in-domain and transfer performance are
    measured; leave `held_out` empty for an all-in-domain baseline.

    The shared data knobs (`context_size`, `prompt_style`) mirror the RL env's
    `load_environment`, so the classifier reads the same episodes/prompts — train on `train`, select
    on `val`, `test` held out; there is no n_eval/eval_seed knob (the split partition is the
    published dataset's). `prompt_style` (direct vs. reasoning) applies to all domains via each env's
    `system_prompt_for`. `max_*_examples` and `shuffle_seed` are trainer-specific
    (the infinite per-rank sampler); `max_*_examples` truncation applies per-domain.

    Eval runs per-domain (`eval/<domain>/...`) over the union, plus two macro-averaged aggregates:
    `eval/in_domain/...` over the trained `domains` and `eval/held_out/...` over `held_out` (the
    latter only when `held_out` is non-empty), so in-distribution and transfer numbers stay separate.

    `output_dir` defaults to `outputs/<domains-joined>-meta-classifier` when unset (for a single domain,
    `outputs/<domain>-meta-classifier`); set it explicitly (or via the
    `OUTPUT_DIR` env var) to override.
    """

    domains: list[DomainSpec] = Field(default_factory=list)
    # Domains to evaluate but NOT train on (cross-domain transfer). Disjoint from `domains`; no weight.
    held_out: list[DomainSpec] = Field(default_factory=list)
    # Backward-compatible scalar form (normalized into `domains` below); prefer `[[domains]]`.
    domain: Literal["gwas", "depmap", "tahoe"] | None = None
    data_path: str | None = None
    output_dir: Path | None = None
    max_steps: int = Field(2500, ge=1)
    eval_interval: int = Field(25, ge=1)
    grad_accum_steps: int = Field(1, ge=1)
    dist_timeout_seconds: int = Field(1800, ge=1)
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    context_size: int = Field(16, ge=0)
    # Per-episode random context length: when set to (lo, hi), each episode renders K ~ Uniform{lo..hi}
    # stubs instead of the fixed `context_size`, keyed by episode_id (+ `context_seed`) so K is stable
    # across runs/splits and matches the RL env's render for the same episode. See
    # ibis_env_utils.context_size_for. Applies to both train and val.
    context_size_range: tuple[int, int] | None = None
    context_seed: int = 0
    # prompt_style (direct vs. reasoning) applies to all domains (each env exposes system_prompt_for).
    prompt_style: Literal["direct", "reasoning"] = "direct"
    max_train_examples: int | None = Field(None, ge=1)
    max_eval_examples: int | None = Field(None, ge=1)
    shuffle_seed: int = 0
    log: TrainerLogConfig = Field(default_factory=TrainerLogConfig)
    wandb: WandbConfig | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimizerConfig = Field(default_factory=AdamWConfig)
    scheduler: SchedulerConfig = Field(default_factory=ConstantSchedulerConfig)
    ckpt: ClassificationCheckpointConfig | None = None

    @model_validator(mode="after")
    def _normalize_domains(self) -> "ClassificationConfig":
        # Accept either the scalar (domain + data_path) form or the [[domains]] list, not both.
        if self.domains and (self.domain is not None or self.data_path is not None):
            raise ValueError("set either scalar `domain`/`data_path` or `[[domains]]`, not both")
        if not self.domains:
            if self.domain is None or self.data_path is None:
                raise ValueError("must set `[[domains]]`, or both scalar `domain` and `data_path`")
            self.domains = [DomainSpec(domain=self.domain, data_path=self.data_path)]
        names = [spec.domain for spec in self.domains]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate domains in `[[domains]]`: {names}")
        weighted = [spec.weight is not None for spec in self.domains]
        if any(weighted) and not all(weighted):
            raise ValueError("`weight` must be set on all domains or none")
        # Held-out domains are evaluated but never trained: disjoint from the trained mix, no weight.
        held_out_names = [spec.domain for spec in self.held_out]
        if len(set(held_out_names)) != len(held_out_names):
            raise ValueError(f"duplicate domains in `[[held_out]]`: {held_out_names}")
        overlap = set(held_out_names) & set(names)
        if overlap:
            raise ValueError(f"held-out domains must not also be trained on: {sorted(overlap)}")
        if any(spec.weight is not None for spec in self.held_out):
            raise ValueError("`weight` is not allowed on `[[held_out]]` domains (they aren't trained)")
        return self

    @model_validator(mode="after")
    def _validate_context_range(self) -> "ClassificationConfig":
        if self.context_size_range is not None:
            lo, hi = self.context_size_range
            if not (0 <= lo <= hi):
                raise ValueError(f"context_size_range must satisfy 0 <= lo <= hi, got {self.context_size_range}")
        return self

    @model_validator(mode="after")
    def _default_output_dir(self) -> "ClassificationConfig":
        if self.output_dir is None:
            tag = "-".join(spec.domain for spec in self.domains)
            self.output_dir = Path(f"outputs/{tag}-meta-classifier")
        return self


# Adapted from the prime-rl SFT trainer's validation pass (prime_rl/trainer/sft/train.py). Metrics (BCE/Brier/accuracy) are task-specific.
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_by_domain: dict[str, list[dict]],
    in_domain: list[str],
    held_out: list[str],
    tokenizer,
    config: ClassificationConfig,
    parallel_dims: ParallelDims,
    data_rank: int,
    data_world_size: int,
) -> dict[str, float]:
    """Per-domain BCE/Brier/accuracy over the val records, reduced across ranks.

    Every rank iterates the same `val_by_domain` (built identically from `config.domains` +
    `config.held_out`, so keys and order match), calling `aggregate` once per domain — keeping the
    cross-rank collectives aligned even when a rank's slice of a domain is empty. Logs
    `eval/<domain>/{loss,brier,brier_reward,accuracy}` per domain, plus two macro-averaged (unweighted
    over domains) aggregates: `eval/in_domain/*` over the trained `in_domain` domains and
    `eval/held_out/*` over the `held_out` domains (emitted only when non-empty). Keeping the two
    separate means a held-out (transfer) domain never contaminates the in-distribution number.
    """
    model.eval()
    pad_to_multiple = parallel_dims.seq_len_divisor
    log_metrics: dict[str, float] = {}
    per_domain: dict[str, dict[str, float]] = {}
    for domain, records in val_by_domain.items():
        totals = {"loss": 0.0, "brier": 0.0, "correct": 0.0, "count": 0.0}
        for record in records[data_rank::data_world_size]:
            input_ids, position_ids, last_token_idx, label = encode_record(
                record,
                tokenizer,
                config.model.seq_len,
                pad_to_multiple,
            )
            logits = classifier_logits(model, input_ids, position_ids, last_token_idx, config, record["forward_kwargs"])
            loss = F.binary_cross_entropy_with_logits(logits, label)
            prob = torch.sigmoid(logits)
            totals["loss"] += loss.item()
            totals["brier"] += ((prob - label) ** 2).item()
            totals["correct"] += ((prob >= 0.5) == (label >= 0.5)).float().item()
            totals["count"] += 1.0

        totals = aggregate(totals)
        count = max(totals["count"], 1.0)
        metrics = {
            "loss": totals["loss"] / count,
            "brier": totals["brier"] / count,
            "brier_reward": 1.0 - totals["brier"] / count,
            "accuracy": totals["correct"] / count,
        }
        for key, value in metrics.items():
            log_metrics[f"eval/{domain}/{key}"] = value
        per_domain[domain] = metrics

    # Macro (unweighted over domains) aggregates, in-domain and held-out kept separate.
    for group, names in (("in_domain", in_domain), ("held_out", held_out)):
        if not names:
            continue
        for key in ("loss", "brier", "brier_reward", "accuracy"):
            log_metrics[f"eval/{group}/{key}"] = sum(per_domain[n][key] for n in names) / len(names)
    return log_metrics


# Recipe adapted from the prime-rl SFT trainer (prime_rl/trainer/sft/train.py). The LM-head -> scalar-head
# swap and the last-token BCE objective are the task-specific changes.
@clean_exit
def train(config: ClassificationConfig) -> None:
    """Train the scalar classification head on the configured domain(s) until max_steps."""
    in_domain = [spec.domain for spec in config.domains]
    held_out = [spec.domain for spec in config.held_out]

    # Distributed + output setup
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Starting classifier trainer: train on {in_domain}, hold out {held_out} in {world}")

    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload or config.model.optim_cpu_offload,
    )
    if output_dir := os.environ.get("OUTPUT_DIR"):
        config.output_dir = Path(output_dir)
    torch.set_float32_matmul_precision(config.matmul_precision)
    if world.is_master:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    parallel_dims = get_parallel_dims(config.model, config.model.seq_len)
    non_data_parallel_size = parallel_dims.non_data_parallel_size
    data_rank = world.rank // non_data_parallel_size
    data_world_size = parallel_dims.get_mesh("dp").size()

    # Data + tokenizer. Train pools come from the trained domains; val pools cover trained + held-out,
    # so the two dicts can have different keys (val includes any held-out domain not trained on).
    train_by_domain, val_by_domain = build_all_records(config)
    for domain, records in train_by_domain.items():
        logger.info(f"Loaded {len(records)} train {domain} examples")
    for domain, records in val_by_domain.items():
        logger.info(f"Loaded {len(records)} val {domain} examples")
    tokenizer = setup_tokenizer(
        TokenizerConfig(name=config.model.name, trust_remote_code=config.model.trust_remote_code)
    )

    # Resolve checkpoint to resume from (if any) before building the model
    loading_from_ckpt_later = False
    ckpt_manager, _ = setup_ckpt_managers(config.output_dir, config.ckpt)
    checkpoint_step = None
    if config.ckpt and ckpt_manager is not None and config.ckpt.resume_step is not None:
        checkpoint_step = (
            resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
            if config.ckpt.resume_step == -1
            else config.ckpt.resume_step
        )
        loading_from_ckpt_later = checkpoint_step is not None

    # Build + shard the model, then swap the LM head for the scalar classification head
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
    hidden_size = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    head_dtype = DTYPE_MAP[config.model.optimization_dtype]
    # setup_fsdp shards the model's lm_head jointly with the final norm as one FSDP group with
    # reshard_after_forward=False. Swapping in the plain scalar head orphans the norm in that group:
    # the old head never runs forward, so the group's pre-hook can't re-unshard it after a
    # checkpoint's get_state_dict reshards it. Keep the handle so save_checkpoint can re-unshard.
    norm_lm_head_group = model.lm_head
    classifier_head = ScalarClassificationHead(hidden_size).to(device="cuda", dtype=head_dtype)
    model.lm_head = classifier_head

    # Optimizer over the FSDP base params, plus the plain-Tensor head as a separate param group
    dtensor_named_parameters, tensor_named_parameters = split_named_parameters_by_type(list(model.named_parameters()))
    optimizer = setup_optimizer(
        config.optim,
        dtensor_named_parameters,
        parallel_dims,
        cpu_offload=config.model.optim_cpu_offload,
    )
    if tensor_named_parameters:
        base_optimizer = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
        base_optimizer.add_param_group({"params": [parameter for _, parameter in tensor_named_parameters]})

    scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    progress = Progress()
    if checkpoint_step is not None:
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        logger.info(f"Resumed from checkpoint step {checkpoint_step}")

    # Equal-weight mixture unless every domain sets an explicit weight (validated all-or-none).
    weights = (
        {spec.domain: spec.weight for spec in config.domains}
        if all(spec.weight is not None for spec in config.domains)
        else None
    )
    sampler = MixtureDataset(train_by_domain, weights, config.shuffle_seed, data_rank, data_world_size)
    monitor = setup_monitor(config.wandb, output_dir=config.output_dir, run_config=config)
    metrics_path = config.output_dir / "classification_metrics.jsonl"

    # Training loop
    while progress.step < config.max_steps:
        start = time.perf_counter()
        model.train()
        classifier_head.train()
        optimizer.zero_grad(set_to_none=True)

        totals = {"loss": 0.0, "brier": 0.0, "correct": 0.0, "tokens": 0.0, "count": 0.0}
        for _ in range(config.grad_accum_steps):
            record = sampler.next()
            input_ids, position_ids, last_token_idx, label = encode_record(
                record,
                tokenizer,
                config.model.seq_len,
                parallel_dims.seq_len_divisor,
            )
            logits = classifier_logits(
                model,
                input_ids,
                position_ids,
                last_token_idx,
                config,
                record["forward_kwargs"],
            )
            # BCE on the last-token logit
            loss = F.binary_cross_entropy_with_logits(logits, label) / config.grad_accum_steps
            loss.backward()

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                totals["loss"] += loss.item() * config.grad_accum_steps
                totals["brier"] += ((prob - label) ** 2).item()
                totals["correct"] += ((prob >= 0.5) == (label >= 0.5)).float().item()
                totals["tokens"] += float(input_ids.shape[1])
                totals["count"] += 1.0

        # The head holds plain Tensors (not FSDP DTensors), so average its grads across ranks manually
        for parameter in classifier_head.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(data_world_size)

        # Clip (DTensor and plain-Tensor params take different clip paths), then step
        if config.optim.max_norm is not None:
            dtensor_parameters, tensor_parameters = split_parameters_by_grad_type(model.parameters())
            if dtensor_parameters:
                clip_grad_norm_(dtensor_parameters, max_norm=config.optim.max_norm, ep_enabled=parallel_dims.ep_enabled)
            if tensor_parameters:
                torch.nn.utils.clip_grad_norm_(tensor_parameters, max_norm=config.optim.max_norm)
        optimizer.step()
        scheduler.step()

        progress.step += 1
        progress.total_samples += int(data_world_size * config.grad_accum_steps)
        progress.total_tokens += int(totals["tokens"] * data_world_size)

        # Metrics, periodic eval, and checkpoint
        metrics = aggregate(totals)
        count = max(metrics["count"], 1.0)
        log_metrics = {
            "train/loss": metrics["loss"] / count,
            "train/brier": metrics["brier"] / count,
            "train/brier_reward": 1.0 - metrics["brier"] / count,
            "train/accuracy": metrics["correct"] / count,
            "train/tokens_per_step": metrics["tokens"] / count,
            "train/step_time_seconds": time.perf_counter() - start,
            "step": progress.step,
        }

        should_stop = False
        if val_by_domain and progress.step % config.eval_interval == 0:
            log_metrics.update(
                evaluate(
                    model,
                    val_by_domain,
                    in_domain,
                    held_out,
                    tokenizer,
                    config,
                    parallel_dims,
                    data_rank,
                    data_world_size,
                )
            )
            # Persist the eval metrics to the per-step snapshot before pruning/early-stop read it.
            # Master writes; the barrier makes the file visible to every rank so best-N pruning
            # (below) and should_early_stop compute the same keep-set / decision on all ranks.
            if world.is_master:
                write_metric_snapshot(config.output_dir, progress.step, log_metrics)
            dist.barrier()

            # Early stopping (single process, so just break the loop — no sentinel/orchestrator).
            # Reuses the shared best_metric/best_mode selection; keep_best_n retains the best ckpt.
            if config.ckpt and config.ckpt.early_stop_patience is not None:
                history = load_metric_history(get_metrics_dir(config.output_dir), config.ckpt.best_metric)
                should_stop = should_early_stop(
                    history,
                    progress.step,
                    config.ckpt.early_stop_patience,
                    config.ckpt.best_mode,
                    config.ckpt.early_stop_min_delta,
                )

        # best-N retention runs inside save_checkpoint (reads the snapshot above).
        if config.ckpt and config.ckpt.interval and progress.step % config.ckpt.interval == 0:
            save_checkpoint(
                ckpt_manager,
                progress.step,
                model,
                optimizer,
                scheduler,
                progress,
                config.output_dir,
                config.ckpt,
                norm_lm_head_group,
            )

        monitor.log(log_metrics, step=progress.step)
        if world.is_master:
            nan = float("nan")
            line = (
                f"step={progress.step} train_reward={log_metrics['train/brier_reward']:.4f} "
                f"eval_reward[in_domain]={log_metrics.get('eval/in_domain/brier_reward', nan):.4f}"
            )
            if held_out:
                line += f" eval_reward[held_out]={log_metrics.get('eval/held_out/brier_reward', nan):.4f}"
            logger.info(line)
            with metrics_path.open("a") as f:
                f.write(json.dumps(log_metrics) + "\n")

        if should_stop:
            logger.info(
                f"Early stopping at step {progress.step}: {config.ckpt.best_metric} has not improved "
                f"for >= {config.ckpt.early_stop_patience} steps"
            )
            break

    # Final checkpoint (captures the last / early-stop step; best-N keeps the best regardless).
    save_checkpoint(
        ckpt_manager,
        progress.step,
        model,
        optimizer,
        scheduler,
        progress,
        config.output_dir,
        config.ckpt,
        norm_lm_head_group,
    )


# Follows the prime-rl trainer entrypoint pattern in prime_rl/trainer/sft/train.py (set_proc_title + cli + train).
def main() -> None:
    """CLI entrypoint: parse config and run training."""
    config = cli(ClassificationConfig)
    tag = "".join(spec.domain.capitalize() for spec in config.domains)
    set_proc_title(f"{tag}Classifier")
    train(config)


if __name__ == "__main__":
    main()
