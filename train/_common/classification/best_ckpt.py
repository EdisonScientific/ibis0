"""Validation-metric checkpointing for the classifier: best-N retention and patience early stopping.

Upstream prime-rl keeps checkpoints by recency only. The paper's classifier runs selected checkpoints
by validation loss instead: each eval writes a per-step metric snapshot
(``<output_dir>/metrics/step_N.json``), the N best checkpoints by ``best_metric`` are retained, and
training stops once the metric has not improved for ``early_stop_patience`` steps. The selection and
patience rules below are the ones those runs used.
"""

import json
import shutil
from pathlib import Path
from typing import Literal

from prime_rl.configs.trainer import CheckpointConfig
from pydantic import Field, model_validator


class ClassificationCheckpointConfig(CheckpointConfig):
    """prime-rl's checkpoint config plus validation-metric selection."""

    keep_best_n: int | None = Field(None, ge=1)
    """Keep only the N best checkpoints by ``best_metric`` (plus the most recent and any not yet scored).
    Mutually exclusive with ``keep_last`` / ``keep_interval``. ``interval`` should be a multiple of the
    eval interval so every checkpoint is scored."""

    best_metric: str = "eval/in_domain/loss"
    """Full metric key (as logged, e.g. ``eval/in_domain/loss``) used to rank checkpoints and judge plateaus."""

    best_mode: Literal["max", "min"] = "min"
    """Whether higher (``max``) or lower (``min``) ``best_metric`` is better."""

    early_stop_patience: int | None = Field(None, ge=1)
    """Stop once ``best_metric`` has not improved for this many steps. None disables early stopping."""

    early_stop_min_delta: float = Field(0.0, ge=0.0)
    """Minimum improvement in ``best_metric`` that resets the patience counter."""

    @model_validator(mode="after")
    def _best_n_excludes_recency(self) -> "ClassificationCheckpointConfig":
        if self.keep_best_n is not None and (self.keep_last is not None or self.keep_interval is not None):
            raise ValueError("ckpt.keep_best_n is mutually exclusive with ckpt.keep_last / ckpt.keep_interval")
        return self


def get_metrics_dir(output_dir: Path) -> Path:
    return output_dir / "metrics"


def get_metric_snapshot_path(output_dir: Path, step: int) -> Path:
    return get_metrics_dir(output_dir) / f"step_{step}.json"


def write_metric_snapshot(path: Path, metrics: dict) -> None:
    """Merge the numeric scalars of ``metrics`` into the per-step snapshot at ``path``, atomically.

    Only ``{full_metric_key: float}`` scalars are kept (``step``, ``*policy_version``, bools and
    non-numeric values are skipped). An empty ``metrics`` still creates the file, so the step is
    recorded as evaluated-but-unscorable rather than pending."""
    scalars = {
        key: float(value)
        for key, value in metrics.items()
        if key != "step"
        and not key.endswith("policy_version")
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(scalars)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing))
    tmp.replace(path)


def load_metric_history(metrics_dir: Path, metric_key: str) -> dict[int, float | None]:
    """Read per-step snapshots into ``{step: metric_or_None}``.

    ``None`` = snapshot exists but the key is missing / non-numeric (evaluated but unscorable); steps
    without a snapshot are omitted (pending)."""
    metrics: dict[int, float | None] = {}
    if not metrics_dir.exists():
        return metrics
    for snapshot in sorted(metrics_dir.glob("step_*.json")):
        try:
            step = int(snapshot.stem.split("_")[-1])
        except ValueError:
            continue
        try:
            data = json.loads(snapshot.read_text())
        except (OSError, ValueError):
            metrics[step] = None
            continue
        value = data.get(metric_key) if isinstance(data, dict) else None
        # bool is an int subclass; a metric is never a flag, so reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            metrics[step] = None
            continue
        metrics[step] = float(value)
    return metrics


def select_best_steps(
    ckpt_steps: list[int],
    step_metrics: dict[int, float | None],
    keep_best_n: int,
    best_mode: str,
    max_pending: int | None = None,
) -> set[int]:
    """Return the subset of ``ckpt_steps`` to keep under best-N-by-metric retention.

    The ``keep_best_n`` best-scored steps are kept (ties toward the more recent step); unscored
    (``None``) steps are evictable; pending steps are kept, up to the newest ``max_pending``; the most
    recent checkpoint is always kept; if nothing is scored yet, everything is kept."""
    steps = sorted(set(ckpt_steps))
    if not steps:
        return set()

    scored = {s: m for s, m in step_metrics.items() if s in steps and m is not None}
    if not scored:
        return set(steps)

    keep: set[int] = set()
    ordered = sorted(scored, reverse=True)
    ordered.sort(key=lambda s: scored[s], reverse=(best_mode == "max"))
    keep.update(ordered[:keep_best_n])

    pending = [s for s in steps if s not in step_metrics]
    if max_pending is not None and len(pending) > max_pending:
        pending = pending[-max_pending:]
    keep.update(pending)

    keep.add(steps[-1])
    return keep


def _best_scored_step(scored: dict[int, float], best_mode: str, min_delta: float) -> int:
    """The step whose metric last improved on the running best by more than ``min_delta``."""
    steps = sorted(scored)
    improved = (
        (lambda new, best: new > best + min_delta) if best_mode == "max" else (lambda new, best: new < best - min_delta)
    )
    best_step = steps[0]
    best_val = scored[best_step]
    for step in steps[1:]:
        if improved(scored[step], best_val):
            best_step, best_val = step, scored[step]
    return best_step


def should_early_stop(
    step_metrics: dict[int, float | None],
    current_step: int,
    patience: int,
    best_mode: str,
    min_delta: float = 0.0,
) -> bool:
    """True once ``current_step`` is at least ``patience`` steps past the last improving step.

    Unscorable steps never count as improvements; returns False until at least one step is scored."""
    scored = {s: m for s, m in step_metrics.items() if m is not None}
    if not scored:
        return False
    return current_step - _best_scored_step(scored, best_mode, min_delta) >= patience


def prune_best_n(ckpt_manager, output_dir: Path, config: ClassificationCheckpointConfig) -> None:
    """Delete checkpoints outside the best-N keep set. Every rank computes the same keep set from
    the shared snapshot files; only the master deletes."""
    steps_to_keep = select_best_steps(
        ckpt_manager.ckpt_steps,
        load_metric_history(get_metrics_dir(output_dir), config.best_metric),
        config.keep_best_n,
        config.best_mode,
        max_pending=config.keep_best_n,
    )
    if ckpt_manager.world.is_master:
        for step in ckpt_manager.ckpt_steps:
            step_dir = ckpt_manager.get_ckpt_path(step).parent
            if step not in steps_to_keep and step_dir.exists():
                shutil.rmtree(step_dir)
    ckpt_manager.ckpt_steps = [s for s in ckpt_manager.ckpt_steps if s in steps_to_keep]
