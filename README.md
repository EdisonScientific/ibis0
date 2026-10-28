# Ibis0: evidence deliberation for scientific hypotheses

Code, data and models for **"Do Language Models Know What Compelling Scientific Evidence Looks Like?"**
([arXiv](ARXIV_URL)).

Ibis0 models read a biological hypothesis together with a set of experimental results (evidence
"stubs" from GWAS, DepMap CRISPR screens and Tahoe-100M perturbation data) and report a calibrated
probability that the hypothesis holds. This repository contains the EDB-Bio data builders, the
reward environments, the training recipes, the test-set evaluation harness, every analysis behind the
paper's tables and figures, and the Bayesian experimental design (BED) experiments.

## Released artifacts

| Artifact | Location | Licence |
|---|---|---|
| EDB-Bio (train/val/test; `gwas`, `depmap`, `tahoe`) | [`EdisonScientific/EDB-Bio`](https://huggingface.co/datasets/EdisonScientific/EDB-Bio) @ `v1.0.0` | CC BY 4.0 |
| Test sets (primary + sensitivity) | `testsets/` in [`EdisonScientific/EDB-Bio`](https://huggingface.co/datasets/EdisonScientific/EDB-Bio) @ `v1.0.0` | CC BY 4.0 |
| Ibis0 non-thinking (RL) | [`EdisonScientific/Ibis0-30B-A3B-NonThinking`](https://huggingface.co/EdisonScientific/Ibis0-30B-A3B-NonThinking) | Apache-2.0 |
| Ibis0 thinking (RL) | [`EdisonScientific/Ibis0-30B-A3B-Thinking`](https://huggingface.co/EdisonScientific/Ibis0-30B-A3B-Thinking) | Apache-2.0 |
| Ibis0 classifier (+ leave-one-domain-out branches) | [`EdisonScientific/Ibis0-30B-A3B-Classifier`](https://huggingface.co/EdisonScientific/Ibis0-30B-A3B-Classifier) | Apache-2.0 |
| BED outcome simulator (θ′) | [`EdisonScientific/Ibis0-BED-Simulator`](https://huggingface.co/EdisonScientific/Ibis0-BED-Simulator) | Apache-2.0 |
| Training logs, per-episode eval outputs | [`EdisonScientific/ibis0-paper-artifacts`](https://huggingface.co/datasets/EdisonScientific/ibis0-paper-artifacts) | CC BY 4.0 |

All models are fine-tuned from [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B).

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). macOS (Apple Silicon or Intel)
and Linux (x86_64 or aarch64) are supported for data preparation, analysis, and API-based
evaluation. Training and local GPU inference require Linux and CUDA 12.8+ GPUs; the
CUDA training dependencies are installed only on Linux.

```bash
git clone --recurse-submodules https://github.com/EdisonScientific/ibis0.git
cd ibis0
# prime-rl's nested submodules use SSH URLs; without GitHub SSH keys, rewrite them first:
#   git config --global url."https://github.com/".insteadOf git@github.com:
uv sync --frozen --all-groups
cp .env.example .env   # then edit: IBIS_DATA, HF_TOKEN, cache locations, API keys
mkdir -p logs
uv run pytest          # ~300 CPU tests, under a minute
```

Training uses [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), pinned as a submodule at
`src/prime_rl` (commit `570463cca`). The lockfile reproduces the paper's environment (torch 2.11,
vLLM 0.22, transformers 5.14).

## Using the models

The RL models are ordinary causal LMs; serve them with vLLM and send the prompts built by the
environments (`envs/<domain>_meta`). The classifier is a sequence-classification model with a
scalar head (`sigmoid(logit)` = P(hypothesis holds)):

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo = "EdisonScientific/Ibis0-30B-A3B-Classifier"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(
    repo, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
model = model.to("cuda").eval()  # ~61 GB of bf16 weights: one 80 GB+ GPU
messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
p = torch.sigmoid(model(input_ids=ids.to(model.device)).logits.float()).item()
```

`scripts/export/predict_classifier.py` scores the published test set this way.

## Repository layout

```
envs/                       reward/eval environments (verifiers): gwas_meta, depmap_meta, tahoe_meta,
                            and ibis_env_utils (dataset loading, splits, shared reward)
train/_common/              shared RL and classifier recipes and launchers
train/_common/classification/  classifier trainer (scalar head on prime-rl's Qwen3-MoE)
train/multi_meta/runs/      the paper's runs (*__r2.1.0): rl-direct, rl-thinking, cls, leave-one-out cls
train/multi_meta/sft/bed-simulator/  θ′ outcome-simulator SFT
scripts/data_pipeline/      EDB-Bio builders (fetch → build → validate → publish)
scripts/analysis/multidomain/  test-set build, evaluation and every table/figure (render_all.sh)
scripts/analysis/rq2_evidence_deliberation/  weight-of-evidence reliability figure
scripts/analysis/rq5_bed/   Bayesian experimental design experiments
scripts/post_training/      Slurm launchers for test-set evaluation
scripts/export/             DCP → Hugging Face classifier export and HF inference
scripts/select_checkpoint.py  validation-based checkpoint selection for RL runs
```

All outputs default to `$IBIS_DATA` (default `./ibis_data`): `runs/`, `eval/`, `raw/`, `bed_sim/`.

## Reproducing the paper

### 1. Data

The released EDB-Bio files are what the paper trained on. To rebuild them from the public sources:

```bash
uv run python scripts/data_pipeline/fetch_gwas_raw.py --help
uv run python scripts/data_pipeline/build_gwas_dataset.py --help
uv run python scripts/data_pipeline/build_depmap_dataset.py --help
sbatch scripts/data_pipeline/build_tahoe_meta_dataset.sbatch
```

See [`scripts/data_pipeline/README.md`](scripts/data_pipeline/README.md). The test sets are built
with `scripts/analysis/multidomain/build_static_testset.py` (byte-identical on rebuild).

### 2. Training

All runs used nodes of 8×B200 (Slurm). Set `partition` in each run's `slurm.toml` /
`#SBATCH --partition` for your cluster.

```bash
# RL (non-thinking and thinking)
train/_common/launch-rl.sh train/multi_meta/runs/gwas-depmap-tahoe__qwen3-30b-a3b__rl-direct__r2.1.0
train/_common/launch-rl.sh train/multi_meta/runs/gwas-depmap-tahoe__qwen3-30b-a3b__rl-thinking__r2.1.0

# Classifier (and the leave-one-domain-out variants in the same directory)
RUN=train/multi_meta/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0
sbatch --job-name="$(basename "$RUN")" train/_common/launch-classification.sbatch "$RUN"

# θ′ BED outcome simulator
sbatch train/multi_meta/sft/bed-simulator/launch-sft.sbatch
```

**Checkpoint selection.** The classifier keeps its two best checkpoints by validation loss and stops
after 1000 steps without improvement (built into its trainer). For RL, the paper kept the two best
checkpoints by mean held-out reward across the three domains and stopped after 1000 steps without
improvement. Upstream prime-rl retains checkpoints by recency only, so the RL configs keep every
checkpoint, stop at the step the paper's runs stopped (4500 non-thinking, 3400 thinking), and
`scripts/select_checkpoint.py` applies the same rule from the eval rollouts:

```bash
uv run python scripts/select_checkpoint.py $IBIS_DATA/runs/<run> --prune --watch   # alongside training
```

Selected checkpoints: non-thinking step 4200, thinking step 2400, classifier step 2400
(leave-one-out: without tahoe 1700, without depmap 1300, without gwas 1800). The exact as-run
configs, including the checkpoint options of the prime-rl version used for the paper, are kept in
each RL run's `as-run/rl.toml`.

RL runs may stall while writing the final checkpoint after `max_steps`. Cancel the job at that point:
interval checkpoints and exported weights are already complete.

Export a classifier checkpoint to Hugging Face format:

```bash
uv run python scripts/export/export_classifier.py $IBIS_DATA/runs/<cls-run>/checkpoints/step_N/trainer outputs/<name>
```

### 3. Evaluation

```bash
# classifier (prime-rl checkpoint, one 8-GPU node)
RUN=train/multi_meta/runs/gwas-depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0
STEP=2400 sbatch --job-name="$(basename "$RUN")-testset" scripts/post_training/eval_classification_testset.sbatch "$RUN"
# RL models and base Qwen3 (vLLM on one GPU)
MODEL=EdisonScientific/Ibis0-30B-A3B-NonThinking MODES=direct sbatch scripts/post_training/eval_generative_testset.sbatch
# frontier models (API)
sbatch scripts/post_training/eval_frontier_testset.sbatch anthropic claude-opus-5
```

Each launcher documents its arguments in its header. Results go to `$IBIS_DATA/eval/<testset@revision>/<model>/`.
The per-episode outputs behind the paper are in the artifacts repository; download them into
`$IBIS_DATA/eval/` to re-render without re-running inference.

### 4. Tables and figures

```bash
bash scripts/analysis/multidomain/render_all.sh     # tables + figures into outputs/
```

See [`scripts/analysis/multidomain/README.md`](scripts/analysis/multidomain/README.md) for the
pipeline and each renderer, and `scripts/analysis/rq5_bed/` for the BED experiments.

## Citation

```bibtex
@article{ibis0,
  title  = {Do Language Models Know What Compelling Scientific Evidence Looks Like?},
  author = {Igoe, Conor and Grand, Gabriel and Koch, Zane and Bou, Albert and Moss, Eli and Hinks, Michaela M. and Rodriques, Samuel G. and White, Andrew and Narayanan, Siddharth},
  journal = {arXiv preprint arXiv:ARXIV_ID},
  year   = {2026}
}
```

## Licence

Code and model weights: Apache-2.0 (see [LICENSE](LICENSE)). Datasets and released logs: CC BY 4.0,
with attribution to the upstream sources listed on each dataset card (GWAS Catalog, DepMap,
Tahoe-100M).
