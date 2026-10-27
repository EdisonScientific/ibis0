"""Training curves: held-out eval-reward vs training step for BOTH RL runs (direct + reasoning), with legend.

Reads {run}/run_default/rollouts/step_*/eval_rollouts*.jsonl (per-episode rewards.reward), plots
mean +/- 1 SE per step. Colors match the reliability figure (direct blue, reasoning green)."""

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument(
    "--runs-base",
    default=os.environ.get("IBIS_RUNS_BASE", "outputs"),
    help="directory that contains the run output dirs (default: $IBIS_RUNS_BASE or ./outputs)",
)
ap.add_argument(
    "--direct-run",
    default="tahoe-qwen30b-a3b-tahoe-direct-32k-512-5000-brier-temp1-eval100-normaladv-keepeval-v2",
    help="run dir name for the direct model, relative to --runs-base",
)
ap.add_argument(
    "--reason-run",
    default="tahoe-qwen30b-a3b-reason-32k-24k-5000-brier-temp1-eval100-normaladv-keepeval-scale6rr-lr5e6-rerun-v1",
    help="run dir name for the reasoning model, relative to --runs-base",
)
ap.add_argument("--out", default="outputs/figures/training_curve", help="output path stem (no extension)")
args = ap.parse_args()
RUNS = [
    ("Ibis0 (non-thinking)", str(Path(args.runs_base) / args.direct_run), "#0072B2"),
    ("Ibis0 (thinking)", str(Path(args.runs_base) / args.reason_run), "#009E73"),
]


def _reward(line):
    d = json.loads(line)
    r = d.get("rewards")
    if isinstance(r, dict) and r.get("reward") is not None:
        return r["reward"]
    return d.get("reward")


def curve(run_dir):
    root = Path(run_dir) / "run_default" / "rollouts"
    paths = sorted(
        glob.glob(str(root / "step_*" / "eval_rollouts*.jsonl")),
        key=lambda p: int(p.split("step_")[1].split("/")[0]),
    )
    steps, means, ses = [], [], []
    for p in paths:
        step = int(p.split("step_")[1].split("/")[0])
        rew = [v for line in open(p) if (v := _reward(line)) is not None]
        if not rew:
            continue
        a = np.asarray(rew, float)
        steps.append(step)
        means.append(a.mean())
        ses.append(a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
    return np.array(steps), np.array(means), np.array(ses)


fig, ax = plt.subplots(figsize=(6, 6))
for label, rd, col in RUNS:
    s, m, se = curve(rd)
    if len(s) == 0:
        print("no eval rollouts for", label)
        continue
    ax.errorbar(
        s,
        m,
        yerr=se,
        marker="o",
        ms=4,
        color=col,
        ecolor="#94a3b8",
        elinewidth=1.0,
        capsize=2.5,
        linewidth=1.8,
        label=label,
    )
    print(f"{label}: {len(s)} eval steps, peak reward {m.max():.3f} @ step {int(s[m.argmax()])}")
ax.set_xlabel("training step")
ax.set_ylabel("held-out eval reward")
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
out = Path(args.out)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=150, bbox_inches="tight")
print("wrote", str(out) + ".pdf/.png")
