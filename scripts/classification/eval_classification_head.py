"""Evaluate a scalar classification-head checkpoint on pre-rendered records — domain-agnostic.

Companion to the classification-head trainer that swaps a base model's LM head for a scalar head and
trains it with last-token BCE. This loads such a checkpoint (prime-rl DCP) and scores a generic JSONL
of records, so any domain can transfer-eval the same model by rendering its own prompts into the
records format.

Records JSONL — one object per line, rendered by the calling domain:
    {"episode_id": "...", "system_prompt": "...", "question": "...", "label": 0|1, ...passthrough...}

The model forward is collective across the model-parallel mesh, so every rank runs its data shard;
per-record probabilities are all-gathered and de-duplicated by episode_id. Writes <out> (per-record
prob/label JSONL) and prints Brier / ECE / AUROC via the shared calibration metrics.

Launch (torchrun / SLURM, same parallelism as training — cp=1 for the classifier):
    uv run torchrun --nproc-per-node 8 scripts/classification/eval_classification_head.py @ config.toml

Serve mode (serve=True): instead of the one-shot batch, keep the model resident and answer belief
queries over an OpenAI-compatible /v1/chat/completions endpoint (rank 0 serves HTTP and broadcasts
each prompt to the other ranks for the collective forward; the reply is sigmoid(prob) wrapped as
`<probability>p</probability>` text). Lets interactive callers — e.g. an adaptive belief/EIG
estimator — drive thousands of data-dependent queries against the same head with no client changes:
    uv run torchrun --nproc-per-node 8 scripts/classification/eval_classification_head.py \
        @ config.toml --serve --serve-port 8000   # then point a caller at http://<node>:8000/v1
"""

import json
import math
import queue
import sys
import threading
import time
from collections.abc import Mapping
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import prime_rl._compat  # noqa: F401
import torch
import torch.distributed as dist
from prime_rl.configs.shared import TrainerLogConfig
from prime_rl.configs.trainer import CheckpointConfig, ModelConfig, TokenizerConfig
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.model import DTYPE_MAP, setup_model, setup_tokenizer
from prime_rl.trainer.parallel_dims import get_parallel_dims
from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import resolve_latest_ckpt_step
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit
from pydantic import Field
from torch import Tensor

# scripts/lib for the shared calibration metrics, plus the classification trainer's module dir so the
# scalar head and record encoder come from one source of truth instead of being re-defined here
# (repo's scripts-root resolution convention).
_scripts_root = next(p for p in Path(__file__).resolve().parents if p.name == "scripts")
sys.path.insert(0, str(_scripts_root / "lib"))
sys.path.insert(0, str(_scripts_root.parent / "train" / "tahoe_meta" / "classification"))
from calibration_metrics import compute_calibration  # noqa: E402
from model_utils import ScalarClassificationHead  # noqa: E402

try:
    from model_utils import encode_record  # noqa: E402
except ImportError:
    from data_adapters import encode_record  # noqa: E402


class EvalConfig(BaseConfig):
    """Eval a classification-head checkpoint over a records JSONL."""

    model: ModelConfig
    log: TrainerLogConfig = TrainerLogConfig()
    # skip_optimizer => model-only load (the classifier checkpoint's optimizer state omits frozen
    # vision params; we only need the weights for eval, so don't reconstruct optimizer/scheduler).
    ckpt: CheckpointConfig = CheckpointConfig(resume_step=-1, skip_optimizer=True)

    run_dir: Path = Field(..., description="training run dir holding checkpoints/ (ckpt manager root)")
    records_path: Path | None = Field(
        None, description="JSONL of {episode_id, system_prompt, question, label} (required unless serve=True)"
    )
    out: Path = Path("outputs/classification/eval_preds.jsonl")
    max_examples: int | None = Field(None, ge=1)
    dist_timeout_seconds: int = 1800
    matmul_precision: str = "high"

    # Serve mode: keep the loaded model resident and answer belief queries over an OpenAI-compatible
    # HTTP endpoint instead of running the one-shot batch eval. Lets interactive callers (e.g. an
    # adaptive belief/EIG estimator) drive thousands of data-dependent belief queries against the
    # same scalar head. The head's forward is collective, so rank 0 serves HTTP and broadcasts each
    # prompt to the other ranks; every rank runs the forward, rank 0 returns sigmoid(prob) as text.
    serve: bool = Field(False, description="Run as a resident belief server instead of a batch eval")
    serve_host: str = "0.0.0.0"
    serve_port: int = 8000
    # Dynamic batching: concurrent HTTP requests are coalesced into one collective forward (batch
    # dim > 1) so the model parallelism is amortized instead of paying it per single-example query.
    serve_max_batch: int = Field(16, ge=1, description="Max requests coalesced into one forward")
    serve_linger_s: float = Field(0.01, ge=0.0, description="Seconds to let a burst accumulate before a batch")


@torch.no_grad()
def classifier_logit(model, input_ids, position_ids, last_token_idx) -> Tensor:
    outputs = model(
        input_ids=input_ids, position_ids=position_ids
    )  # prime-rl custom Qwen3-MoE forward() takes no use_cache
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    return logits[:, last_token_idx, 0].float()


def _encode_batch(messages, tokenizer, max_seq_len, pad_to_multiple):
    """Tokenize a list of (system, question) pairs into one right-padded batch.

    Right padding is safe for a last-token classifier head under causal attention: each row's final
    real token attends only to earlier (real) positions, never the trailing pads, so its logit equals
    the unbatched case. Every rank calls this on the SAME broadcast messages and so builds identical
    tensors, keeping the collective forward in lockstep.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs = []
    for system, question in messages:
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": question}],
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
            raise ValueError(f"record renders to {len(ids)} tokens > max_seq_len={max_seq_len}")
        seqs.append(ids)
    real_lens = [len(s) for s in seqs]
    padded_len = max(math.ceil(rl / pad_to_multiple) * pad_to_multiple for rl in real_lens)
    bsz = len(seqs)
    input_ids = torch.full((bsz, padded_len), pad_id, dtype=torch.long, device="cuda")
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device="cuda")
    position_ids = torch.arange(padded_len, dtype=torch.long, device="cuda").unsqueeze(0).repeat(bsz, 1)
    last_idx = torch.tensor([rl - 1 for rl in real_lens], dtype=torch.long, device="cuda")
    return input_ids, position_ids, last_idx


def _tokenize_one(system, question, tokenizer, max_seq_len):
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
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
        ids = ids[:max_seq_len]
    return ids


@torch.no_grad()
def _serve_forward_batch(model, tokenizer, seq_len, pad_to_multiple, messages) -> list:
    """VARLEN-PACKED batched forward. The prime-rl custom Qwen3-MoE uses varlen attention keyed off
    position_ids resetting to 0 per sequence; a naive [bsz, padded_len] batch violates that contract
    and triggers an illegal memory access. So we PACK messages into one [1, total] sequence (each with
    its own 0..len-1 position_ids), greedily up to seq_len tokens per forward, and read each message's
    last-token logit — exactly the scheme annotate_stub_woe uses. Returns one sigmoid prob per message."""
    seqs = [_tokenize_one(s, q, tokenizer, seq_len) for (s, q) in messages]
    probs: list = [0.5] * len(seqs)
    p = 0
    while p < len(seqs):
        ids_flat, pos_flat, ends, members = [], [], [], []
        while p < len(seqs):
            L = len(seqs[p])
            if members and len(ids_flat) + L > seq_len:
                break
            ids_flat.extend(seqs[p])
            pos_flat.extend(range(L))
            ends.append(len(ids_flat) - 1)
            members.append(p)
            p += 1
            if len(ids_flat) >= seq_len:
                break
        input_ids = torch.tensor([ids_flat], dtype=torch.long, device="cuda")
        position_ids = torch.tensor([pos_flat], dtype=torch.long, device="cuda")
        outputs = model(input_ids=input_ids, position_ids=position_ids)
        logits = (outputs["logits"] if isinstance(outputs, dict) else outputs.logits)[0, :, 0].float()
        pr = torch.sigmoid(logits[torch.tensor(ends, dtype=torch.long, device="cuda")]).tolist()
        for k, v in zip(members, pr):
            probs[k] = v
    return probs


def _serve_worker(model, tokenizer, seq_len, pad_to_multiple, logger) -> None:
    """Non-master ranks: block on each broadcast, run the collective batched forward in lockstep,
    discard the results (only rank 0 returns them over HTTP). Exits when rank 0 broadcasts a stop."""
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0)
        msg = box[0]
        if not msg or msg.get("cmd") == "stop":
            logger.info("serve worker received stop")
            return
        try:
            _serve_forward_batch(model, tokenizer, seq_len, pad_to_multiple, msg["msgs"])
        except Exception as e:  # symmetric with the master's batch worker; keep the rank alive
            logger.warning(f"serve worker batch forward failed: {e}")


class _BatchJob:
    """One in-flight belief query: its prompt plus an Event the batch worker sets once the result
    (or error) is ready for the waiting HTTP handler thread."""

    __slots__ = ("system", "question", "prob", "error", "done")

    def __init__(self, system, question):
        self.system = system
        self.question = question
        self.prob = None
        self.error = None
        self.done = threading.Event()


def _serve_master(config, model, tokenizer, seq_len, pad_to_multiple, logger) -> None:
    """Rank 0: serve an OpenAI-compatible /v1/chat/completions endpoint with dynamic batching.

    HTTP handler threads enqueue jobs and block on a per-job Event. A single batch-worker thread
    drains the queue (up to serve_max_batch, after a short linger so a concurrent burst coalesces),
    broadcasts the batch to all ranks, runs ONE collective forward, and routes each prob back. The
    worker is the sole issuer of collectives on rank 0, staying in lockstep with the other ranks'
    single broadcast-receive per loop.
    """
    model_id = str(config.model.name)
    jobs: queue.Queue = queue.Queue()
    stop = _BatchJob(None, None)  # sentinel pushed at shutdown to release the batch worker

    def batch_worker():
        while True:
            first = jobs.get()
            if first is stop:
                dist.broadcast_object_list([{"cmd": "stop"}], src=0)
                return
            if config.serve_linger_s > 0:
                time.sleep(config.serve_linger_s)  # let a concurrent burst land before draining
            batch = [first]
            while len(batch) < config.serve_max_batch:
                try:
                    nxt = jobs.get_nowait()
                except queue.Empty:
                    break
                if nxt is stop:  # honor shutdown, but serve what we already drained first
                    jobs.put(stop)
                    break
                batch.append(nxt)
            msgs = [(j.system, j.question) for j in batch]
            dist.broadcast_object_list([{"cmd": "forward", "msgs": msgs}], src=0)
            try:
                probs = _serve_forward_batch(model, tokenizer, seq_len, pad_to_multiple, msgs)
                for j, p in zip(batch, probs):
                    j.prob = p
            except Exception as e:  # symmetric: the other ranks ran the same forward and also raised
                logger.warning(f"serve batch forward failed (bsz={len(batch)}): {e}")
                for j in batch:
                    j.error = str(e)
            for j in batch:
                j.done.set()

    worker_thread = threading.Thread(target=batch_worker, name="batch-worker", daemon=True)
    worker_thread.start()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # silence per-request stderr spam
            pass

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._json(200, {"object": "list", "data": [{"id": model_id, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                messages = body["messages"]
                system = next(m["content"] for m in messages if m["role"] == "system")
                question = next(m["content"] for m in messages if m["role"] == "user")
            except Exception as e:
                self._json(400, {"error": f"bad request: {e}"})
                return
            job = _BatchJob(system, question)
            jobs.put(job)
            if not job.done.wait(timeout=config.dist_timeout_seconds):
                self._json(503, {"error": "batch worker timeout"})
                return
            if job.error is not None:
                self._json(500, {"error": f"forward failed: {job.error}"})
                return
            content = f"<probability>{job.prob:.6f}</probability>"
            self._json(
                200,
                {
                    "id": "chatcmpl-clshead",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", model_id),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            )

    server = ThreadingHTTPServer((config.serve_host, config.serve_port), Handler)
    logger.info(
        f"classification-head belief server on {config.serve_host}:{config.serve_port} "
        f"(model={model_id}, max_batch={config.serve_max_batch}, linger={config.serve_linger_s}s)"
    )
    try:
        server.serve_forever()
    finally:
        jobs.put(stop)  # release the batch worker, which broadcasts stop to the other ranks
        worker_thread.join(timeout=30)


@clean_exit
def evaluate(config: EvalConfig) -> None:
    set_proc_title("classification-eval")
    world = get_world()
    logger = setup_logger(config.log.level, json_logging=config.log.json_logging)
    logger.info(f"Classification-head eval in {world}")
    # Mirror the trainer's init: CPU-offloaded params need gloo for the CPU-side collectives, and the
    # checkpoint was written under the same setting, so match it for a consistent sharded load.
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload or config.model.optim_cpu_offload,
    )
    torch.set_float32_matmul_precision(config.matmul_precision)

    parallel_dims = get_parallel_dims(config.model, config.model.seq_len)
    non_dp = parallel_dims.non_data_parallel_size
    data_rank = world.rank // non_dp
    data_world_size = parallel_dims.get_mesh("dp").size()

    records = []
    if not config.serve:
        if config.records_path is None:
            raise ValueError("records_path is required unless serve=True")
        records = [json.loads(line) for line in config.records_path.read_text().splitlines() if line.strip()]
        if config.max_examples:
            records = records[: config.max_examples]
        if not records:
            raise ValueError(
                f"{config.records_path} has no records to score; the metrics are undefined on an empty set"
            )
        ids = [r.get("episode_id") for r in records]
        if any(i is None for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(
                "every record needs a unique 'episode_id'; it keys the all-gather de-dup, so a missing "
                "or duplicate id would silently collapse distinct records into one"
            )
        logger.info(f"Loaded {len(records)} records from {config.records_path}")

    tokenizer = setup_tokenizer(
        TokenizerConfig(name=config.model.name, trust_remote_code=config.model.trust_remote_code)
    )

    # Build + shard the base model, swap LM head for the scalar head, then load the checkpoint.
    step = (
        resolve_latest_ckpt_step(config.run_dir / "checkpoints")
        if config.ckpt.resume_step == -1
        else config.ckpt.resume_step
    )
    model = setup_model(config.model, parallel_dims, loading_from_checkpoint_later=True)
    hidden_size = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    model.lm_head = ScalarClassificationHead(hidden_size).to(
        device="cuda", dtype=DTYPE_MAP[config.model.optimization_dtype]
    )

    # Model-only load (skip_optimizer set on the config): no optimizer/scheduler/progress needed for eval.
    ckpt_manager, _ = setup_ckpt_managers(config.run_dir, config.ckpt)
    ckpt_manager.load(step, model, [], None, None)
    logger.info(f"Loaded classification checkpoint step {step} from {config.run_dir}")

    model.eval()
    pad_to_multiple = parallel_dims.seq_len_divisor

    if config.serve:
        if world.is_master:
            _serve_master(config, model, tokenizer, config.model.seq_len, pad_to_multiple, logger)
        else:
            _serve_worker(model, tokenizer, config.model.seq_len, pad_to_multiple, logger)
        return

    shard = records[data_rank::data_world_size]
    # The model forward is collective across the data-parallel mesh, so every rank must run the same
    # number of forwards. When the record count is not divisible by data_world_size the shards differ
    # in length by one, so pad each to the global max with a filler forward whose result is discarded.
    # Otherwise a short-shard rank reaches the all_gather below while the others are still inside a
    # collective forward, and the job deadlocks until the dist timeout.
    n_iters = math.ceil(len(records) / data_world_size)
    local = []
    for i in range(n_iters):
        record = shard[i] if i < len(shard) else records[0]  # records[0]: filler, dropped below
        input_ids, position_ids, last_idx, label = encode_record(
            record, tokenizer, config.model.seq_len, pad_to_multiple
        )
        prob = torch.sigmoid(classifier_logit(model, input_ids, position_ids, last_idx)).item()
        if world.is_master and (i + 1) % 25 == 0:
            logger.info(f"scored {i + 1}/{n_iters} iterations")
        if i >= len(shard):
            continue  # filler forward run only to stay lockstep with longer shards; not a real record
        # Pass through every record key except the rendering inputs (kept out of the output to avoid
        # echoing the bulky prompt text), then attach the computed prob and the normalized label.
        passthrough = {k: v for k, v in record.items() if k not in ("system_prompt", "question", "label")}
        local.append({**passthrough, "prob": prob, "label": int(record["label"])})

    gathered = [None] * world.world_size
    dist.all_gather_object(gathered, local)
    if not world.is_master:
        return

    by_id, ordered = {}, []
    for chunk in gathered:
        for rec in chunk:
            key = rec["episode_id"]
            if key not in by_id:
                by_id[key] = rec
                ordered.append(rec)
    # Gather walks rank chunks, so `ordered` follows the shard interleave; restore records_path order
    # (episode_ids are unique) so the preds JSONL stays line-aligned with the input.
    original_order = {rec["episode_id"]: i for i, rec in enumerate(records)}
    ordered.sort(key=lambda rec: original_order[rec["episode_id"]])

    probs = np.array([r["prob"] for r in ordered])
    labels = np.array([r["label"] for r in ordered])
    rep = compute_calibration(probs, labels).to_dict()
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("w") as f:
        for r in ordered:
            f.write(json.dumps(r) + "\n")
    summary = {
        "n": len(ordered),
        "brier": round(rep["brier"], 4),
        "ece": round(rep["ece"], 4),
        "auroc": round(rep["auroc"], 4) if rep.get("auroc") is not None else None,
        "mean_prob": round(float(probs.mean()), 4),
        "base_rate": round(float(labels.mean()), 4),
    }
    config.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[classification-eval] {summary}")


def main():
    evaluate(cli(EvalConfig))


if __name__ == "__main__":
    main()
