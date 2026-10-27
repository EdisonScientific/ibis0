"""Validation-metric checkpoint selection for prime-rl RL runs (best-N retention + patience).

The paper's RL runs kept the 2 best checkpoints by mean eval reward across the eval environments
(``eval/all/avg@4``) and stopped once it had not improved for 1000 steps. Upstream prime-rl keeps
checkpoints by recency only, so this script recomputes that metric from the eval rollouts the
orchestrator writes (``<run>/run_default/rollouts/step_N/eval_rollouts_<env>.jsonl``) and applies
the same selection and patience rules as the classifier (``train/_common/classification/best_ckpt.py``).

    # report the selected step and whether the patience rule has fired
    uv run python scripts/select_checkpoint.py <run-output-dir>
    # also delete checkpoints/weights outside the keep set; --watch re-runs every --interval seconds
    uv run python scripts/select_checkpoint.py <run-output-dir> --prune --watch

The metric for a step is the mean over eval envs of each env's mean rollout reward; with the same
number of rollouts per example this equals each env's ``avg@k``, macro-averaged as ``eval/all/avg@k``.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train" / "_common" / "classification"))
from best_ckpt import select_best_steps, should_early_stop  # noqa: E402


def step_metrics(run_dir: Path, envs: list[str] | None) -> dict[int, float | None]:
    """``{step: macro-mean eval reward}`` for every step with eval rollouts; ``None`` if any env is missing."""
    metrics: dict[int, float | None] = {}
    for step_dir in (run_dir / "run_default" / "rollouts").glob("step_*"):
        files = sorted(step_dir.glob("eval_rollouts_*.jsonl"))
        if not files:
            continue
        per_env = {}
        for f in files:
            env = f.stem.removeprefix("eval_rollouts_")
            rewards = [json.loads(line)["reward"] for line in f.read_text().splitlines() if line.strip()]
            per_env[env] = sum(float(r) for r in rewards) / len(rewards) if rewards else None
        wanted = envs or sorted(per_env)
        values = [per_env.get(e) for e in wanted]
        metrics[int(step_dir.name.split("_")[-1])] = None if None in values else sum(values) / len(values)
    return metrics


def checkpoint_steps(run_dir: Path) -> list[int]:
    steps = set()
    for sub in ("checkpoints", "weights"):
        steps.update(int(p.name.split("_")[-1]) for p in (run_dir / sub).glob("step_*") if p.is_dir())
    return sorted(steps)


def run_once(args) -> bool:
    metrics = step_metrics(args.run_dir, args.envs)
    scored = {s: m for s, m in metrics.items() if m is not None}
    if not scored:
        print("no scored eval steps yet")
        return False
    # Patience is measured from the trainer's step count, not the last eval: rollout batches are
    # numbered from 0, so the latest batch step_N means N + 1 completed steps.
    current = 1 + max(int(p.name.split("_")[-1]) for p in (args.run_dir / "run_default" / "rollouts").glob("step_*"))
    best = (max if args.mode == "max" else min)(scored, key=lambda s: (scored[s], s))
    stop = should_early_stop(metrics, current, args.patience, args.mode, args.min_delta)
    print(
        f"evaluated steps: {len(metrics)} (training step {current}); best step {best} = {scored[best]:.4f}; "
        f"patience reached: {stop}"
    )
    if args.prune:
        ckpts = checkpoint_steps(args.run_dir)
        keep = select_best_steps(ckpts, metrics, args.keep_best_n, args.mode, max_pending=args.keep_best_n)
        for step in ckpts:
            if step in keep:
                continue
            for sub in ("checkpoints", "weights"):
                path = args.run_dir / sub / f"step_{step}"
                if path.exists():
                    print(f"removing {path}")
                    shutil.rmtree(path)
        print(f"kept checkpoint steps: {sorted(keep)}")
    return stop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="the RL run's output directory (passed as --output-dir)")
    ap.add_argument("--envs", nargs="+", default=None, help="eval env names to average (default: all found)")
    ap.add_argument("--mode", choices=["max", "min"], default="max")
    ap.add_argument("--keep-best-n", type=int, default=2)
    ap.add_argument("--patience", type=int, default=1000, help="steps without improvement before stopping")
    ap.add_argument("--min-delta", type=float, default=0.001, help="improvement that resets patience")
    ap.add_argument("--prune", action="store_true", help="delete checkpoints/weights outside the keep set")
    ap.add_argument("--watch", action="store_true", help="repeat until the patience rule fires")
    ap.add_argument("--interval", type=int, default=600, help="seconds between --watch passes")
    args = ap.parse_args()

    while not run_once(args) and args.watch:
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
