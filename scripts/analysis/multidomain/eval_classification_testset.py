"""Published-test-set eval for a BCE classification-head checkpoint, scoring EVERY eval domain under a
single model load.

Emits per-episode probs in the generative sweep's JSON format so the classifier joins every figure
(grouped_metrics, reliability, dose, WoE, table) alongside the generative models.

Two design points:

1. **Input is a published test set** (`testsets/` in `EdisonScientific/EDB-Bio`), where every row is
   already one forward pass — `context_pool` pre-truncated, `k`/`n_stubs` resolved. There is no episode
   sampling, no seed, and no context flags here: the set *is* the definition, identical for every model
   in the roster. A `primary` config yields one pass; a `sensitivity` config yields one pass per `k`.
2. **Multi-domain in one invocation.** The checkpoint is domain-independent, so the model is loaded once
   and each requested domain is scored under that load; only `build_record` / `system_prompt_for` /
   `forward_kwargs` are resolved per domain. That matters because the load is a sharded 30B DCP read
   under torchrun — a leave-one-out row of the transfer matrix (one model, three domains) becomes a
   single allocation instead of three.

Env vars (the config supplies model/ckpt/parallelism as usual):
  SWEEP_OUT_DIR     result directory for THIS model (REQUIRED), from
                    scripts/lib/eval_paths.result_dir(). Cells land at `<dir>/<domain>_<tag>.json`,
                    where tag is "primary" or "ctx<k>". The launcher owns the layout and writes run.json.
  SWEEP_TESTSET_REV Hub revision of an hf:// SWEEP_TESTSET (the launcher passes the pinned TESTSET_REV)
  SWEEP_TESTSET     published test-set path template containing `{domain}` (REQUIRED), e.g.
                    "hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}". A local
                    directory of the same shape works too — `load_dataset_split` treats the two
                    identically, which is what makes a cheap local smoke set possible.
  SWEEP_DOMAINS     comma list of eval domains to score under one load (REQUIRED), e.g.
                    "gwas,depmap,tahoe".

Run via the same torchrun/SLURM pattern as the trainer (loads under training parallelism).
"""

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

# scripts/lib is not on the launcher's PYTHONPATH (which carries the trainer + env modules), so add it
# the same way the generative and frontier evals do — `testset_passes` MUST be the one shared definition
# or two roster rows silently score different episode sets under the same column name.
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "scripts") / "lib"))

import numpy as np
import prime_rl._compat  # noqa: F401
import torch
import torch.distributed as dist
from data_adapters import encode_record, get_adapter
from ibis_env_utils import Split, load_dataset_split
from model_utils import ScalarClassificationHead, classifier_logits, split_named_parameters_by_type
from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.model import DTYPE_MAP, setup_model, setup_tokenizer
from prime_rl.trainer.optim import setup_optimizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.scheduler import setup_scheduler
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit
from testset_io import testset_passes
from train_classification_head import ClassificationConfig


def _testset_passes(df) -> list[tuple[str, list[dict]]]:
    """Thin alias for the shared reader — see scripts/lib/testset_io.testset_passes.

    A local copy would be the failure mode testset_io exists to prevent: if
    the classifier's notion of "which episodes make up a pass" ever drifted from the generative and
    frontier evals', the roster rows would stop being comparable while still reporting the same column.
    """
    return testset_passes(df)


def _resolve_eval_targets(domains_env: str | None, testset: str | None) -> list[tuple[str, str]]:
    """The [(domain, data_path), ...] scored under one model load.

    Both env vars are required and neither falls back to the config: the whole point of a published set
    is that the eval targets are named explicitly rather than inferred from a training mix.
    """
    if not domains_env:
        raise ValueError("set SWEEP_DOMAINS=<d1,d2,...> — the eval domains to score.")
    if not testset:
        raise ValueError("set SWEEP_TESTSET=<published test-set path template containing '{domain}'>.")
    if "{domain}" not in testset:
        raise ValueError(f"SWEEP_TESTSET must contain '{{domain}}' to expand per domain, got {testset!r}")
    domains = [d.strip() for d in domains_env.split(",") if d.strip()]
    if not domains:
        raise ValueError("SWEEP_DOMAINS is set but empty.")
    targets = [(d, testset.format(domain=d)) for d in domains]
    # A fully expanded path must contain no braces. Catches a template that smuggled in a stray brace —
    # e.g. bash's ${TESTSET:-...{domain}} mis-parse, which appends "}" to an overridden value. Cheap to
    # check here, and it fails in seconds instead of after a multi-minute 30B checkpoint load.
    bad = [(d, p) for d, p in targets if "{" in p or "}" in p]
    if bad:
        raise ValueError(f"expanded test-set path still contains braces (bad SWEEP_TESTSET template?): {bad}")
    return targets


@torch.no_grad()
def _score(model, records, tokenizer, config, parallel_dims, data_rank, data_world_size, forward_kwargs):
    """Per-episode scalar-head probs over this rank's shard; returns [{episode_id,domain,label,parsed_prob,n_stubs}].

    The model is FSDP-sharded, so **every forward is a collective** (a parameter all-gather). All data
    ranks must therefore run the SAME number of forwards: an uneven shard deadlocks, because the ranks
    holding an extra record wait on an all-gather that the short rank never joins — it has already moved
    on to `all_gather_object`. NCCL busy-waits, so this presents as every GPU pinned at 100% with no
    progress rather than as an error.

    Records are padded by cycling to a multiple of `data_world_size`. The duplicates are free: the
    master rank merges results into an `episode_id`-keyed dict, which collapses them.

    Padding matters whenever an episode count is not a multiple of the data world size, which nothing
    about a test set or a parallelism config guarantees.
    """
    model.eval()
    pad = parallel_dims.seq_len_divisor
    if records and (extra := (-len(records)) % data_world_size):
        records = records + [records[i % len(records)] for i in range(extra)]
    out = []
    for rec in records[data_rank::data_world_size]:
        input_ids, position_ids, last_idx, label = encode_record(rec, tokenizer, config.model.seq_len, pad)
        logit = classifier_logits(model, input_ids, position_ids, last_idx, config, forward_kwargs)
        out.append(
            {
                "episode_id": rec["episode_id"],
                "domain": rec.get("domain"),
                "label": int(rec["label"]),
                "parsed_prob": float(torch.sigmoid(logit).item()),
                "n_stubs": rec["n_stubs"],
            }
        )
    return out


@clean_exit
def sweep(config: ClassificationConfig) -> None:
    import importlib

    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    out_dir = Path(os.environ["SWEEP_OUT_DIR"])
    targets = _resolve_eval_targets(os.environ.get("SWEEP_DOMAINS"), os.environ.get("SWEEP_TESTSET"))

    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload or config.model.optim_cpu_offload,
    )
    if output_dir := os.environ.get("OUTPUT_DIR"):
        config.output_dir = Path(output_dir)
    torch.set_float32_matmul_precision(config.matmul_precision)

    parallel_dims = get_parallel_dims(config.model, config.model.seq_len)
    non_data_parallel_size = parallel_dims.non_data_parallel_size
    data_rank = world.rank // non_data_parallel_size
    data_world_size = parallel_dims.get_mesh("dp").size()

    assert config.ckpt is not None and config.ckpt.resume_step is not None and config.ckpt.resume_step >= 0, (
        "Pass --ckpt.resume-step <STEP>."
    )
    checkpoint_step = config.ckpt.resume_step

    # --- load the model + checkpoint ONCE; it is domain-independent ---
    tokenizer = setup_tokenizer(
        TokenizerConfig(name=config.model.name, trust_remote_code=config.model.trust_remote_code)
    )
    ckpt_manager, _ = setup_ckpt_managers(config.output_dir, config.ckpt)
    assert ckpt_manager is not None
    model = setup_model(config.model, parallel_dims, True)
    hidden_size = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    classifier_head = ScalarClassificationHead(hidden_size).to(
        device="cuda", dtype=DTYPE_MAP[config.model.optimization_dtype]
    )
    model.lm_head = classifier_head
    dtensor_params, tensor_params = split_named_parameters_by_type(list(model.named_parameters()))
    optimizer = setup_optimizer(config.optim, dtensor_params, parallel_dims, cpu_offload=config.model.optim_cpu_offload)
    if tensor_params:
        base_opt = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
        base_opt.add_param_group({"params": [p for _, p in tensor_params]})
    _ = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    progress = Progress()
    ckpt_manager.load(checkpoint_step, model, [optimizer], None, progress)  # scheduler=None (see test eval)
    logger.info(
        f"Loaded checkpoint step {checkpoint_step}; scoring domains "
        f"{[d for d, _ in targets]} (prompt_style={config.prompt_style})"
    )

    # --- score every eval domain under that single load ---
    for eval_domain, eval_data_path in targets:
        # Same public surface on every <domain>_meta module, so one loop serves all domains.
        _env = importlib.import_module(f"{eval_domain}_meta")
        build_record, system_prompt_for = _env.build_record, _env.system_prompt_for
        forward_kwargs = get_adapter(eval_domain).model_forward_kwargs
        sysprompt = system_prompt_for(config.prompt_style)  # classifier convention: 'direct'

        df = load_dataset_split(eval_data_path, Split.TEST, revision=os.environ.get("SWEEP_TESTSET_REV") or None)
        if "testset" not in df.columns:
            raise ValueError(
                f"{eval_data_path} is not a published test set (no `testset` column); build one with "
                "build_static_testset.py."
            )
        passes = _testset_passes(df)
        logger.info(f"[{eval_domain}] {eval_data_path}: {len(df)} rows -> {len(passes)} pass(es)")

        for tag, prows in passes:
            records = []
            for r in prows:
                # build_record slices context_pool[:n]; published rows are pre-truncated, so passing
                # the ACTUAL pool length makes this a no-op rather than a second truncation.
                n_stubs = len(r["context_pool"])
                if int(r["n_stubs"]) != n_stubs:
                    raise ValueError(
                        f"{eval_domain}/{r['episode_id']}: published n_stubs={r['n_stubs']} disagrees "
                        f"with pool length ({n_stubs}) — test set and loader are out of sync."
                    )
                br = build_record(r, n_stubs)
                records.append(
                    {
                        "episode_id": br["info"]["episode_id"],
                        "domain": eval_domain,
                        "question": br["question"],
                        "label": br["info"]["label"],
                        "system_prompt": sysprompt,
                        "n_stubs": n_stubs,
                    }
                )
            local = _score(model, records, tokenizer, config, parallel_dims, data_rank, data_world_size, forward_kwargs)
            gathered: list = [None] * world.world_size
            dist.all_gather_object(gathered, local)
            if world.is_master:
                merged = {}
                for part in gathered:
                    for row in part or []:
                        merged[row["episode_id"]] = row  # dedup cp-duplicated episodes
                results = list(merged.values())
                # Keyed by domain so one job's per-domain outputs cannot collide.
                path = str(out_dir / f"{eval_domain}_{tag}.json")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(
                    json.dumps(
                        {
                            "model": f"classifier:{config.model.name}",
                            "checkpoint_step": checkpoint_step,
                            "domain": eval_domain,
                            "testset": eval_data_path,
                            "context_size": tag,
                            "split": "test",
                            "n_eval": len(results),
                            "results": results,
                        }
                    )
                )
                probs = np.array([r["parsed_prob"] for r in results])
                labs = np.array([r["label"] for r in results])
                brier = float(np.mean((probs - labs) ** 2)) if len(probs) else float("nan")
                logger.info(
                    f"  [{eval_domain}] {tag}: n={len(results)}  Brier={brier:.4f}  "
                    f"mean_p={probs.mean():.3f}  -> {path}"
                )
            dist.barrier()


def main() -> None:
    set_proc_title("ClassifierTestsetEval")
    sweep(cli(ClassificationConfig))


if __name__ == "__main__":
    main()
