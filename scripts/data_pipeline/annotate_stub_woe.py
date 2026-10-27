"""Annotate evidence-deliberation stubs with belief-derived support + weight-of-evidence (WoE).

For each episode in a domain's TRAIN split, run the trained scalar belief model (the `cls` head) over
every consecutive context prefix ``k = 0..K`` (``K = min(pool_len, K_MAX)``). The incremental weight
of evidence contributed by stub ``i`` (whose information set is ``context_pool[:i]``) is

    WoE_i = (logit(p_{i+1}) - logit(p_i)) * 10/ln(10)   decibans,   support_i = WoE_i > 0

with ``p_k = P(supported | context_pool[:k])`` from the belief head. This is the offline annotation
step for the BED conditional-simulator SFT: the simulator later conditions on / emits
``(support, woe)`` alongside the raw observation ``y``.

Reuses the classifier checkpoint-load plumbing from
``scripts/analysis/multidomain/eval_classification_testset.py`` (same DCP load, same
per-episode render as training: ``build_record(row, k)`` + ``system_prompt_for(prompt_style)``).

Scoring is BATCHED: because attention is purely causal and padding is on the right, a right-padded
batch yields the correct last-real-token logit for every row with no attention mask, so we can score
many prefixes per forward. Records are length-sorted then chunked to keep padding waste low.

K_MAX defaults to 16 — the belief model's training ``context_size_range`` upper bound; scoring deeper
prefixes is out-of-distribution for it. For domains with longer pools (tahoe: 32) the deeper stubs go
unlabeled; the count dropped is logged (never silently truncated).

Run once per domain (a single eval domain, like the sweep). Example:
    uv run torchrun --standalone --nproc-per-node 8 scripts/data_pipeline/annotate_stub_woe.py \
        @ train/_common/classification-base.toml \
        --domain gwas --data-path hf://EdisonScientific/EDB-Bio/gwas \
        --output-dir ${IBIS_DATA:-ibis_data}/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0 \
        --ckpt.resume-step 2400

Env vars:
  ANNOT_OUT           output jsonl (default: ibis_data/bed_sim/woe/<domain>.jsonl)
  ANNOT_KMAX          max prefix length / stubs labeled per episode (default 16)
  ANNOT_BATCH         forward batch size (default 8)
  ANNOT_MAX_EPISODES  optional cap on episodes (smoke tests; default: all)
"""

import importlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

# Self-contained sys.path: the classification helpers (data_adapters/model_utils/
# train_classification_head) are plain modules under train/_common/classification, and the env
# modules (+ tahoe's sibling `episodes`) live under envs/<domain>_meta. Add them so this script runs
# regardless of external PYTHONPATH (mirrors eval_classification_head.py's scripts-root convention).
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "train" / "_common").is_dir())
sys.path.insert(0, str(_REPO / "train" / "_common" / "classification"))
for _m in ("ibis_env_utils", "gwas_meta", "depmap_meta", "tahoe_meta"):
    sys.path.insert(0, str(_REPO / "envs" / _m))

import prime_rl._compat  # noqa: F401,E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from ibis_env_utils import Split, load_dataset_split  # noqa: E402
from model_utils import ScalarClassificationHead, split_named_parameters_by_type  # noqa: E402
from prime_rl.configs.trainer import TokenizerConfig  # noqa: E402
from prime_rl.trainer.ckpt import setup_ckpt_managers  # noqa: E402
from prime_rl.trainer.model import DTYPE_MAP, setup_model, setup_tokenizer  # noqa: E402
from prime_rl.trainer.optim import setup_optimizer  # noqa: E402
from prime_rl.trainer.parallel_dims import get_parallel_dims  # noqa: E402
from prime_rl.trainer.runs import Progress  # noqa: E402
from prime_rl.trainer.scheduler import setup_scheduler  # noqa: E402
from prime_rl.trainer.utils import setup_torch_distributed  # noqa: E402
from prime_rl.trainer.world import get_world  # noqa: E402
from prime_rl.utils.config import cli  # noqa: E402
from prime_rl.utils.logger import setup_logger  # noqa: E402
from prime_rl.utils.process import set_proc_title  # noqa: E402
from prime_rl.utils.utils import clean_exit  # noqa: E402
from train_classification_head import ClassificationConfig  # noqa: E402

DECIBAN = 10.0 / math.log(10.0)


def logit(p: float) -> float:
    """Natural-log odds, clipped to [1e-3, 1-1e-3]."""
    p = min(max(p, 1e-3), 1.0 - 1e-3)
    return math.log(p / (1.0 - p))


def _resolve_eval_target(config: ClassificationConfig) -> tuple[str, str]:
    """The single (domain, data_path) to annotate — scalar form or a single-entry [[domains]]."""
    if config.domain is not None and config.data_path is not None:
        return config.domain, config.data_path
    if len(config.domains) == 1:
        return config.domains[0].domain, config.domains[0].data_path
    raise ValueError(
        "annotate_stub_woe scores a single domain; pass --domain <d> --data-path <path> "
        f"(got [[domains]]={[s.domain for s in config.domains]})."
    )


def _encode_ids(question: str, system_prompt: str, tokenizer, max_seq_len: int) -> list[int]:
    """Chat-template a (system, user) pair to token ids — identical settings to the classifier
    trainer's encode_record (add_generation_prompt=True, enable_thinking=False)."""
    from collections.abc import Mapping

    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if len(ids) > max_seq_len:
        raise ValueError(f"render is {len(ids)} tokens > max_seq_len={max_seq_len}")
    return ids


@torch.no_grad()
def _score_packed(model, recs_ids: list[list[int]]) -> list[float]:
    """Pack several records into ONE sequence [1, total] and return P(supported) at each record's last
    token. prime-rl's Qwen3-MoE uses varlen (cu_seqlens) attention derived from ``position_ids``: a
    reset to 0 marks a new segment, so packed records attend only within themselves. This is the
    native batching path (batch-dim [N, L] is unsupported by that forward)."""
    ids: list[int] = []
    pos: list[int] = []
    last_idx: list[int] = []
    off = 0
    for x in recs_ids:
        ids.extend(x)
        pos.extend(range(len(x)))
        last_idx.append(off + len(x) - 1)
        off += len(x)
    input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
    position_ids = torch.tensor([pos], dtype=torch.long, device="cuda")
    out = model(input_ids=input_ids, position_ids=position_ids)
    logits = (out["logits"] if isinstance(out, dict) else out.logits)[0, :, 0].float()  # [total]
    return torch.sigmoid(logits[torch.tensor(last_idx, device="cuda")]).tolist()


# ---- theta_prime (per-stub LOCAL hypothesis) helpers -------------------------------------------
# The candidate experiment x directly measures its own hypothesis theta'. We score P(theta'|{x,y}) by
# swapping the belief model's TARGET to the stub's own (direction-agnostic) hypothesis and using the
# stub itself as the sole evidence. Clean subset (also belief-model in-distribution): gwas (all),
# depmap crispr, tahoe evidence — the modalities whose stubs measure the same quantity render_target
# asserts, so we can reuse the domain's render_target on a remapped row.
def _in_theta_prime_subset(domain, stub):
    if domain == "depmap":
        return stub.get("modality") == "crispr"
    if domain == "tahoe":
        return (stub.get("kind") or "evidence") == "evidence"
    return True  # gwas: every stub is an association, same shape as the target


def _theta_prime_row(domain, r, stub):
    """A minimal row whose target_* fields make render_target emit the stub's OWN hypothesis theta'
    (direction-agnostic: 'is associated with' / 'is a dependency' / 'perturbs')."""
    base = {"episode_id": r["episode_id"], "label": 0, "ood_group": None}
    if domain == "gwas":
        sm = stub.get("snp_metadata") or {}
        base.update(
            target_snp=stub.get("snp"),
            target_trait=stub.get("trait"),
            target_gene=stub.get("gene"),
            target_snp_meta=sm,
            target_gene_meta=stub.get("gene_metadata"),
            target_trait_meta=stub.get("trait_metadata") or {},
            target_direction=None,
            target_allele=sm.get("risk_allele"),
        )
    elif domain == "depmap":
        base.update(
            target_gene=stub.get("gene"),
            target_cell=stub.get("cell"),
            target_gene_meta=stub.get("gene_metadata"),
            target_cell_meta=stub.get("cell_metadata"),
        )
    else:  # tahoe evidence
        dm = stub.get("drug_metadata") or {}
        cm = stub.get("cell_metadata") or {}
        base.update(
            target_drug=dm.get("drug"),
            target_cell=cm.get("cell_name"),
            target_gene=(stub.get("y") or {}).get("gene"),
            target_direction="perturbs",
            target_drug_meta=dm,
            target_cell_meta=cm,
            target_gene_meta=stub.get("gene_metadata") or {},
        )
    return base


@clean_exit
def annotate(config: ClassificationConfig) -> None:
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    domain, data_path = _resolve_eval_target(config)
    env = importlib.import_module(f"{domain}_meta")
    build_record, system_prompt_for = env.build_record, env.system_prompt_for

    k_max = int(os.environ.get("ANNOT_KMAX", "16"))
    # "incremental": WoE of stub i vs its natural prefix I=pool[:i] (context-dependent).
    # "singleton": prior-invariant BF of each stub alone vs the empty prior, WoE_i =
    #   logit(belief({stub_i})) - logit(belief(empty)). Use for the context-free (no info set) simulator.
    mode = os.environ.get("ANNOT_MODE", "incremental")
    split = Split(os.environ.get("ANNOT_SPLIT", "train"))
    max_episodes = os.environ.get("ANNOT_MAX_EPISODES")
    max_episodes = int(max_episodes) if max_episodes else None
    out_path = Path(
        os.environ.get("ANNOT_OUT", str(_REPO / "ibis_data" / "bed_sim" / "woe" / f"{domain}.{split.value}.jsonl"))
    )
    shard_dir = out_path.parent / f"_shards_{domain}_{split.value}"

    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload or config.model.optim_cpu_offload,
    )
    if output_dir := os.environ.get("OUTPUT_DIR"):
        config.output_dir = Path(output_dir)
    torch.set_float32_matmul_precision(config.matmul_precision)

    parallel_dims = get_parallel_dims(config.model, config.model.seq_len)
    non_dp = parallel_dims.non_data_parallel_size
    data_rank = world.rank // non_dp
    data_world_size = parallel_dims.get_mesh("dp").size()
    assert config.ckpt is not None and config.ckpt.resume_step is not None and config.ckpt.resume_step >= 0, (
        "Pass --ckpt.resume-step <STEP> (e.g. 2400)."
    )
    sysprompt = system_prompt_for(config.prompt_style)  # classifier convention: 'direct'

    # ---- load episodes for the requested split; build the flat (episode, k=0..K) record list ----
    df = load_dataset_split(data_path, split)
    rows = df.to_dict("records")
    if max_episodes is not None:
        rows = rows[:max_episodes]
    SOLO = 1_000_000  # key offset marking a "stub alone" record (singleton + theta_prime solo)
    EMPTY = 2_000_000  # key offset marking a per-stub empty-prior record (theta_prime only)
    total_dropped = 0
    flat = []  # (episode_id, k, n_stubs, question, label); k = prefix len, or SOLO+i for a solo stub
    for r in rows:
        pool = r.get("context_pool")
        pool_len = 0 if pool is None else len(pool)
        K = min(pool_len, k_max)
        total_dropped += max(0, pool_len - K)
        if mode == "singleton":
            # empty prior + each stub alone -> prior-invariant per-stub BF
            br0 = build_record(r, 0)
            flat.append((br0["info"]["episode_id"], 0, pool_len, br0["question"], int(br0["info"]["label"])))
            for i in range(K):
                row_i = {**r, "context_pool": [pool[i]]}
                br = build_record(row_i, 1)
                flat.append((br["info"]["episode_id"], SOLO + i, pool_len, br["question"], int(br["info"]["label"])))
        elif mode == "theta_prime":
            # each stub scored against its OWN hypothesis theta': WoE = logit P(theta'|{stub}) - logit P(theta'|empty)
            for i in range(K):
                stub = pool[i]
                if not _in_theta_prime_subset(domain, stub):
                    continue
                trow = _theta_prime_row(domain, r, stub)
                be = build_record({**trow, "context_pool": []}, 0)
                flat.append((be["info"]["episode_id"], EMPTY + i, pool_len, be["question"], int(be["info"]["label"])))
                bs = build_record({**trow, "context_pool": [stub]}, 1)
                flat.append((bs["info"]["episode_id"], SOLO + i, pool_len, bs["question"], int(bs["info"]["label"])))
        else:
            for k in range(K + 1):  # nested prefixes 0..K -> incremental WoE of stubs 0..K-1
                br = build_record(r, k)
                flat.append((br["info"]["episode_id"], k, pool_len, br["question"], int(br["info"]["label"])))
    if world.is_master:
        logger.info(
            f"[{domain}] mode={mode}: {len(rows)} episodes -> {len(flat)} records "
            f"(K_MAX={k_max}); stubs beyond K_MAX dropped: {total_dropped}"
        )

    # ---- load the belief model + scalar head (mirror eval_classification_testset.py) ----
    tokenizer = setup_tokenizer(
        TokenizerConfig(name=config.model.name, trust_remote_code=config.model.trust_remote_code)
    )
    ckpt_manager, _ = setup_ckpt_managers(config.output_dir, config.ckpt)
    assert ckpt_manager is not None
    model = setup_model(config.model, parallel_dims, True)
    hidden_size = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    model.lm_head = ScalarClassificationHead(hidden_size).to(
        device="cuda", dtype=DTYPE_MAP[config.model.optimization_dtype]
    )
    dtensor_params, tensor_params = split_named_parameters_by_type(list(model.named_parameters()))
    optimizer = setup_optimizer(config.optim, dtensor_params, parallel_dims, cpu_offload=config.model.optim_cpu_offload)
    if tensor_params:
        base_opt = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer
        base_opt.add_param_group({"params": [p for _, p in tensor_params]})
    _ = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    progress = Progress()
    ckpt_manager.load(config.ckpt.resume_step, model, [optimizer], None, progress)
    model.eval()
    if world.is_master:
        logger.info(f"[{domain}] loaded belief checkpoint step {config.ckpt.resume_step}")

    # ---- score this rank's shard, LOCKSTEP across ranks ----
    # The model is FSDP-sharded, so every forward is a collective (param all-gather) that ALL ranks
    # must enter together. Per-rank greedy packing yields different pack counts per rank, so if a fast
    # rank stops calling forward while others continue, the stragglers' all-gather deadlocks (NCCL
    # timeout). Fix: pre-build each rank's packs, agree on max pack count via all-reduce, and run
    # exactly that many forwards on every rank — ranks that run out replay their last pack as a
    # discarded dummy so the collectives stay in lockstep.
    shard_dir.mkdir(parents=True, exist_ok=True)
    my = flat[data_rank::data_world_size]
    encoded = [
        (_encode_ids(q, sysprompt, tokenizer, config.model.seq_len), eid, k, n, lab) for (eid, k, n, q, lab) in my
    ]
    encoded.sort(key=lambda e: len(e[0]))  # length-sort so each pack holds similar-length records
    pack_tokens = int(os.environ.get("ANNOT_PACK_TOKENS", str(config.model.seq_len)))
    packs, i = [], 0
    while i < len(encoded):
        pack, tot = [], 0
        while i < len(encoded):
            ln = len(encoded[i][0])
            if pack and tot + ln > pack_tokens:  # a lone over-budget record still forwards alone
                break
            pack.append(encoded[i])
            tot += ln
            i += 1
            if tot >= pack_tokens:
                break
        packs.append(pack)
    if not packs:  # empty shard (never at real scale) — a tiny dummy keeps this rank in lockstep
        packs = [[([0, 1, 2], "__dummy__", 0, 0, 0)]]
    n_rounds_t = torch.tensor([len(packs)], device="cuda")
    dist.all_reduce(n_rounds_t, op=dist.ReduceOp.MAX)
    n_rounds = int(n_rounds_t.item())
    if world.is_master:
        logger.info(
            f"[{domain}] scoring {len(encoded)} records in {n_rounds} lockstep rounds (rank0 has {len(packs)} packs)"
        )

    shard_path = shard_dir / f"rank{world.rank}.jsonl"
    n_done = 0
    with open(shard_path, "w") as fh:
        for r in range(n_rounds):
            real = r < len(packs)
            pack = packs[r] if real else packs[-1]  # replay last pack as a discarded dummy when padding
            probs = _score_packed(model, [c[0] for c in pack])
            if real:
                for (ids, eid, k, n, lab), p in zip(pack, probs):
                    fh.write(json.dumps({"episode_id": eid, "k": k, "prob": p, "n_stubs": n, "label": lab}) + "\n")
                n_done += len(pack)
            if world.is_master and r % 50 == 0:
                logger.info(f"[{domain}] round {r}/{n_rounds} (rank0 scored {n_done}/{len(encoded)})")
    dist.barrier()

    # ---- master: merge shards, compute per-stub WoE, write final jsonl ----
    if world.is_master:
        by_ep: dict[str, dict] = defaultdict(dict)  # episode_id -> {k: prob}
        meta: dict[str, dict] = {}
        for sp in sorted(shard_dir.glob("rank*.jsonl")):
            for line in open(sp):
                d = json.loads(line)
                by_ep[d["episode_id"]][d["k"]] = d["prob"]
                meta[d["episode_id"]] = {"n_stubs": d["n_stubs"], "label": d["label"]}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_rows = 0
        woe_vals = []
        with open(out_path, "w") as fh:
            for eid, kp in by_ep.items():
                if mode == "singleton":
                    p_empty = kp.get(0)
                    if p_empty is None:
                        continue
                    for key, p1 in kp.items():
                        if key < SOLO:
                            continue
                        i = key - SOLO
                        woe = (logit(p1) - logit(p_empty)) * DECIBAN  # prior-invariant BF of stub i
                        woe_vals.append(woe)
                        fh.write(
                            json.dumps(
                                {
                                    "episode_id": eid,
                                    "domain": domain,
                                    "stub_index": i,
                                    "I_len": 0,
                                    "n_stubs": meta[eid]["n_stubs"],
                                    "episode_label": meta[eid]["label"],
                                    "p_before": p_empty,
                                    "p_after": p1,
                                    "woe_decibans": woe,
                                    "support": bool(woe > 0.0),
                                }
                            )
                            + "\n"
                        )
                        n_rows += 1
                elif mode == "theta_prime":
                    for key, p1 in kp.items():
                        if not (SOLO <= key < EMPTY):  # iterate solo records; pair with per-stub empty prior
                            continue
                        i = key - SOLO
                        p0 = kp.get(EMPTY + i)
                        if p0 is None:
                            continue
                        woe = (logit(p1) - logit(p0)) * DECIBAN  # prior-invariant BF of stub i for its OWN theta'
                        woe_vals.append(woe)
                        fh.write(
                            json.dumps(
                                {
                                    "episode_id": eid,
                                    "domain": domain,
                                    "stub_index": i,
                                    "I_len": 0,
                                    "n_stubs": meta[eid]["n_stubs"],
                                    "episode_label": meta[eid]["label"],
                                    "p_before": p0,
                                    "p_after": p1,
                                    "woe_decibans": woe,
                                    "support": bool(woe > 0.0),
                                }
                            )
                            + "\n"
                        )
                        n_rows += 1
                else:
                    K = max(kp)
                    for i in range(K):  # stub i uses prefixes i and i+1
                        if i not in kp or (i + 1) not in kp:
                            continue
                        p0, p1 = kp[i], kp[i + 1]
                        woe = (logit(p1) - logit(p0)) * DECIBAN
                        woe_vals.append(woe)
                        fh.write(
                            json.dumps(
                                {
                                    "episode_id": eid,
                                    "domain": domain,
                                    "stub_index": i,
                                    "I_len": i,
                                    "n_stubs": meta[eid]["n_stubs"],
                                    "episode_label": meta[eid]["label"],
                                    "p_before": p0,
                                    "p_after": p1,
                                    "woe_decibans": woe,
                                    "support": bool(woe > 0.0),
                                }
                            )
                            + "\n"
                        )
                        n_rows += 1
        import numpy as np

        w = np.array(woe_vals) if woe_vals else np.array([0.0])
        logger.info(
            f"[{domain}] wrote {n_rows} stub labels -> {out_path}\n"
            f"  WoE decibans: mean={w.mean():.2f} median={np.median(w):.2f} "
            f"p05={np.percentile(w, 5):.2f} p95={np.percentile(w, 95):.2f} "
            f"min={w.min():.2f} max={w.max():.2f}; support(+)={100 * (w > 0).mean():.1f}%"
        )
    dist.barrier()


def main() -> None:
    set_proc_title("AnnotateStubWoE")
    annotate(cli(ClassificationConfig))


if __name__ == "__main__":
    main()
