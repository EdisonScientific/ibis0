#!/usr/bin/env python
"""Consolidate per-cell eval JSON into one long-format trace table for Bayes-factor analysis.

Walks an eval tree ``<root>/<testset>/<model_key>/{domain}_{primary,ctx<k>}.json`` (the layout defined
in ``scripts/lib/eval_paths.py``) and emits one row per (model_key, domain, cell, episode). The dense
``ctx0..ctx16`` cells written by the sensitivity sweep are what a per-stub Bayes factor needs: for a
single episode, order its rows by ``n_stubs`` (the ACTUAL, pool-capped stub count) and take

    BF_i = odds(p_i) / odds(p_{i-1}),   odds(p) = p / (1 - p)

so the weight of evidence added by stub i is ``10 * log10(BF_i)`` decibans. With the prior-0.5 prompt
the k=0 cell should sit at p≈0.5 (prior odds 1), so the running product of BFs is the posterior odds
directly — see ``traces/README.md`` written alongside the parquet.

Usage:
    uv run --no-sync python scripts/analysis/multidomain/consolidate_traces.py \
        --testset-dir ibis_data/eval/EDB-Bio@df43070d \
        --prompt-tag <tag>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# model_key substring -> the paper's variant label. Order matters only in that the substrings are
# mutually exclusive across the roster, so the first hit is the answer.
_VARIANT = (
    ("__cls__", "classifier"),
    ("__rl-direct__", "non-thinking"),
    ("__rl-thinking__", "thinking"),
)

_TRACE_COLUMNS = [
    "prompt_tag",
    "testset",
    "model_key",
    "variant",
    "model_family",
    "prompt_style",
    "checkpoint_step",
    "domain",
    "cell",
    "context_size",
    "split",
    "episode_id",
    "n_stubs",
    "label",
    "parsed_prob",
]


def variant_of(model_key: str) -> str:
    for pat, name in _VARIANT:
        if pat in model_key:
            return name
    return "other"


def _read_json(path: Path) -> dict | None:
    """Parse ``path`` as JSON, returning None on a missing/unreadable/malformed file.

    An eval tree can hold model dirs this user cannot read or cells that are incomplete; skipping them
    is preferable to aborting the whole consolidation.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_tree(testset_dir: Path, prompt_tag: str) -> pd.DataFrame:
    """One row per (model_key, domain, cell, episode) across every readable model dir under ``testset_dir``."""
    rows: list[dict] = []
    skipped = 0
    for mk_dir in sorted(p for p in testset_dir.iterdir() if p.is_dir()):
        model_key = mk_dir.name
        try:
            cells = sorted(mk_dir.glob("*_*.json"))
        except OSError:
            skipped += 1
            continue
        prov = _read_json(mk_dir / "run.json") or {}
        prompt_style = prov.get("prompt_style")
        model_family = prov.get("model_family")
        for cell in cells:
            if cell.name == "run.json":
                continue
            obj = _read_json(cell)
            if not isinstance(obj, dict) or "results" not in obj:
                continue
            domain = obj.get("domain")
            # cell tag = filename minus the "<domain>_" prefix -> "primary" or "ctx<k>".
            tag = cell.stem[len(domain) + 1 :] if domain and cell.stem.startswith(f"{domain}_") else cell.stem
            for r in obj["results"]:
                rows.append(
                    {
                        "prompt_tag": prompt_tag,
                        "testset": testset_dir.name,
                        "model_key": model_key,
                        "variant": variant_of(model_key),
                        "model_family": model_family,
                        "prompt_style": prompt_style,
                        "checkpoint_step": obj.get("checkpoint_step"),
                        "domain": domain,
                        "cell": tag,
                        "context_size": obj.get("context_size"),
                        "split": obj.get("split"),
                        "episode_id": r.get("episode_id"),
                        "n_stubs": r.get("n_stubs"),
                        "label": r.get("label"),
                        "parsed_prob": r.get("parsed_prob"),
                    }
                )
    if skipped:
        print(f"  (skipped {skipped} unreadable model dir(s))")
    df = pd.DataFrame(rows, columns=_TRACE_COLUMNS)
    return df


_README = """# Bayes-factor traces

`all_traces.parquet` is one row per (model_key, domain, cell, episode), pooled from every
`<model_key>/{domain}_*.json` cell in this eval tree.

Columns: `prompt_tag, testset, model_key, variant, model_family, prompt_style, checkpoint_step,
domain, cell, context_size, split, episode_id, n_stubs, label, parsed_prob`.

- `variant` is `classifier` / `non-thinking` / `thinking`.
- `cell` is `primary` (2k episodes at training K) or `ctx<k>` (the dense sensitivity sweep, 256-episode
  subset, k = 0..16). **The BF sweep lives in the `ctx*` cells.**
- `n_stubs` is the ACTUAL, pool-capped stub count on that row — key BF math off this, not `context_size`.
- `parsed_prob` is the model's p(supported); for the generative rows it is the mean over `n_samples`.

## Per-stub Bayes factor

For one episode, take its `ctx*` rows, sort by `n_stubs`, and

    odds(p) = p / (1 - p)
    BF_i    = odds(p_i) / odds(p_{i-1})            # evidence added by going n_stubs i-1 -> i
    w_i     = 10 * log10(BF_i)                     # weight of evidence, decibans (additive)

With the prior-0.5 prompt the k=0 row sits at p≈0.5 (prior odds 1), so the running product of BFs from
k=0 is the posterior odds itself.

```python
import pandas as pd, numpy as np
df = pd.read_parquet("all_traces.parquet")
sweep = df[df.cell.str.startswith("ctx")].copy()
sweep["odds"] = sweep.parsed_prob.clip(1e-6, 1 - 1e-6).pipe(lambda p: p / (1 - p))
sweep = sweep.sort_values(["variant", "domain", "episode_id", "n_stubs"])
g = sweep.groupby(["variant", "domain", "episode_id"], sort=False)
sweep["bf_step"] = g.odds.transform(lambda s: s / s.shift(1))
sweep["woe_db_step"] = 10 * np.log10(sweep["bf_step"])
```
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--testset-dir",
        required=True,
        type=Path,
        help="the <root>/<testset> directory holding <model_key>/ subdirs (e.g. $IBIS_DATA/eval/EDB-Bio@df43070d)",
    )
    ap.add_argument("--prompt-tag", default="", help="label for this prompt variant, stamped on every row")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output parquet (default: <testset-dir>/traces/all_traces.parquet)",
    )
    args = ap.parse_args()

    if not args.testset_dir.is_dir():
        raise SystemExit(f"not a directory: {args.testset_dir}")

    df = load_tree(args.testset_dir, args.prompt_tag)
    if df.empty:
        raise SystemExit(f"no result cells found under {args.testset_dir}")

    out = args.out or (args.testset_dir / "traces" / "all_traces.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    (out.parent / "README.md").write_text(_README)

    n_sweep = int(df.cell.str.startswith("ctx").sum())
    print(f"wrote {len(df):,} rows -> {out}")
    print(f"  models   : {df.model_key.nunique()}  ({', '.join(sorted(df.variant.unique()))})")
    print(f"  domains  : {', '.join(sorted(x for x in df.domain.dropna().unique()))}")
    print(f"  cells    : {df.cell.nunique()} distinct  ({n_sweep:,} sweep rows for BF)")
    print(f"  episodes : {df.episode_id.nunique():,} distinct")
    print(f"  README   : {out.parent / 'README.md'}")


if __name__ == "__main__":
    main()
