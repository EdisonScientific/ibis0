from pathlib import Path
from typing import TYPE_CHECKING

import prime_rl._compat  # noqa: F401
import torch
import torch.distributed as dist
from best_ckpt import ClassificationCheckpointConfig, get_metric_snapshot_path, prune_best_n
from best_ckpt import write_metric_snapshot as write_metric_snapshot_file
from prime_rl.trainer.runs import Progress
from prime_rl.utils.act_offloading import maybe_activation_offloading
from torch import Tensor, nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from train_classification_head import ClassificationConfig


class ScalarClassificationHead(nn.Module):
    """Linear head mapping the model's hidden state to a single logit."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: Tensor,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> dict:
        _ = labels, temperature
        return {"logits": self.proj(hidden_states)}


# Adapted from the prime-rl SFT trainer forward (prime_rl/trainer/sft/train.py). The last-token logit slice is task-specific.
def classifier_logits(
    model: nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    last_token_idx: int,
    config: "ClassificationConfig",
    forward_kwargs: dict | None = None,
) -> Tensor:
    """Run the model and return the last real token's scalar logit.

    `forward_kwargs` carries any per-domain extra kwargs for the model forward, supplied per record
    by the data adapter (see data_adapters.py). Empty for all current adapters — the shared Qwen3-MoE
    `forward()` accepts no extra kwargs (passing e.g. `use_cache` raises `TypeError`).
    """
    with maybe_activation_offloading(config.model.ac_offloading):
        outputs = model(input_ids=input_ids, position_ids=position_ids, **(forward_kwargs or {}))
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    return logits[:, last_token_idx, 0].float()


# Adapted from the prime-rl SFT trainer's inline cross-rank dist.all_reduce reductions (prime_rl/trainer/sft/train.py).
def aggregate(values: dict[str, float]) -> dict[str, float]:
    """Sum a dict of scalars across all ranks (all-reduce), returning the reduced dict."""
    keys = sorted(values)
    tensor = torch.tensor([values[k] for k in keys], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: tensor[i].item() for i, key in enumerate(keys)}


# Custom. prime-rl shards the base model as FSDP DTensors, but our scalar head is a plain Tensor.
# These two helpers split the parameters so each set gets the right grad-clip / optimizer path.
# prime-rl's own trainers shard the whole model, so they never need this.
def split_parameters_by_grad_type(parameters) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split parameters with a gradient into (DTensor-grad, plain-Tensor-grad) lists."""
    dtensor_parameters = []
    tensor_parameters = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if isinstance(parameter.grad, DTensor):
            dtensor_parameters.append(parameter)
        else:
            tensor_parameters.append(parameter)
    return dtensor_parameters, tensor_parameters


def split_named_parameters_by_type(
    named_parameters: list[tuple[str, nn.Parameter]],
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """Split (name, param) pairs into (DTensor, plain-Tensor) lists."""
    dtensor_parameters = []
    tensor_parameters = []
    for name, parameter in named_parameters:
        if isinstance(parameter, DTensor):
            dtensor_parameters.append((name, parameter))
        else:
            tensor_parameters.append((name, parameter))
    return dtensor_parameters, tensor_parameters


def write_metric_snapshot(output_dir: Path, step: int, metrics: dict[str, float]) -> None:
    """Write this step's metric snapshot (``<output_dir>/metrics/step_N.json``), read back by best-N
    retention and early stopping (``best_ckpt``). Master-only; carries ``eval/in_domain/loss`` and the
    other ``eval/*`` keys the classifier logs."""
    write_metric_snapshot_file(get_metric_snapshot_path(output_dir, step), metrics)


# Adapted from the prime-rl SFT trainer checkpoint save in prime_rl/trainer/sft/train.py (save -> mark_stable -> maybe_clean).
def save_checkpoint(
    ckpt_manager,
    step: int,
    model,
    optimizer,
    scheduler,
    progress: Progress,
    output_dir: Path,
    ckpt_config: ClassificationCheckpointConfig,
    restore_unsharded: FSDPModule | None = None,
) -> None:
    """Persist a checkpoint via the prime-rl checkpoint manager (no-op if disabled).

    `restore_unsharded` is an FSDP group that the save's `get_state_dict` reshards but whose forward
    pre-hook can no longer re-unshard it — specifically the final `[lm_head, norm]` group, orphaned
    when the trainer swaps in the plain scalar head (the old head never runs forward, so the group's
    unshard hook stays disarmed). We re-unshard it here so the next forward sees full params again;
    see train_classification_head for the full rationale.
    """
    if ckpt_manager is None:
        return
    ckpt_manager.save(step, model, [optimizer], scheduler, progress)
    ckpt_manager.mark_stable(step)
    if ckpt_config.keep_best_n is not None:
        prune_best_n(ckpt_manager, output_dir, ckpt_config)
    else:
        ckpt_manager.maybe_clean()
    if isinstance(restore_unsharded, FSDPModule):
        restore_unsharded.unshard()
