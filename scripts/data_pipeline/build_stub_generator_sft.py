"""Build the SFT dataset for the BED stub-value generator.

Task: given a target hypothesis, a REVEALED set of fully-shown evidence stubs, and a candidate
stub's DESCRIPTOR (perturbation/cell/gene identity, relation to target) with its measured VALUES
hidden, predict the candidate's DESeq2 result as JSON. At BED inference time we draw several samples
of these values, complete the stub, and feed each to the belief model to estimate EIG.

Input is a single episodes parquet carrying a `split` column (not the per-split files published to
EDB-Bio). We keep split=="train" rows and drop any episode ids listed in --exclude-ids, so the
generator never sees an episode used to evaluate the belief model (no leakage).

Output: a HuggingFace-loadable parquet (train-00000-of-00001.parquet) in --out-dir, in `messages`
format (system / user / assistant); the SFT loss masks everything but the assistant completion.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

REL = {
    "same_drug_diff_cell": "same drug, different cell line",
    "same_moa_evidence": "different drug, same mechanism of action",
    "diff_drug_same_cell": "different drug and mechanism, same cell line",
    "negative_control": "unrelated drug and cell line (control)",
}

SYSTEM = (
    "You are a model of perturbation biology. Given a target hypothesis and a set of revealed "
    "differential-expression measurements of a gene, you predict the DESeq2 result of a further "
    "candidate experiment on the same gene. Respond with a single JSON object and nothing else."
)

VAL_KEYS = ["log2FoldChange", "lfcSE", "padj", "baseMean", "n_cells_trt", "n_cells_ctrl"]


def drug_desc(meta):
    moa = (meta or {}).get("moa-broad") or "unclear"
    appr = "approved" if (meta or {}).get("human-approved") == "yes" else "not approved"
    return f"{(meta or {}).get('drug', '?')} (MOA: {moa}; {appr})"


def fmt_vals(y):
    def num(k):
        v = y.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if k in ("n_cells_trt", "n_cells_ctrl"):
            return int(v)
        if k == "padj":
            return float(f"{float(v):.2e}")
        return round(float(v), 3)

    return {k: num(k) for k in VAL_KEYS}


def revealed_line(idx, stub):
    y = dict(stub.get("y") or {})
    g = (stub.get("gene_metadata") or {}).get("gene_symbol", "?")
    cell = (stub.get("cell_metadata") or {}).get("cell_name", "?")
    v = fmt_vals(y)
    return (
        f"[{idx}] {drug_desc(stub.get('drug_metadata'))}, cell {cell}, gene {g} "
        f"[{REL.get(stub.get('context_type'), stub.get('context_type'))}]: "
        f"log2FC {v['log2FoldChange']}, lfcSE {v['lfcSE']}, padj {v['padj']}, "
        f"baseMean {v['baseMean']}, n {v['n_cells_trt']}/{v['n_cells_ctrl']}"
    )


def build_example(row, candidate, revealed, rng):
    g = row["target_gene"]
    lines = [
        f"TARGET HYPOTHESIS: {row['target_drug']} {row['target_direction']} {g} in cell line {row['target_cell']}.",
        "",
        f"REVEALED EVIDENCE (DESeq2 measurements of {g} under other perturbations):",
    ]
    if revealed:
        lines += [revealed_line(i + 1, s) for i, s in enumerate(revealed)]
    else:
        lines.append("(none revealed yet)")
    ccell = (candidate.get("cell_metadata") or {}).get("cell_name", "?")
    lines += [
        "",
        "CANDIDATE EXPERIMENT (result hidden):",
        f"{drug_desc(candidate.get('drug_metadata'))}, cell {ccell}, gene {g} "
        f"[{REL.get(candidate.get('context_type'), candidate.get('context_type'))}]",
        "",
        "Predict the DESeq2 result for the candidate experiment as a JSON object with keys "
        + ", ".join(VAL_KEYS)
        + ".",
    ]
    completion = json.dumps(fmt_vals(dict(candidate.get("y") or {})))
    return {
        "messages": [
            {"role": "user", "content": SYSTEM + "\n\n" + "\n".join(lines)},
            {"role": "assistant", "content": completion},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="ibis_data/raw/tahoe-meta-data/v2/dataset.parquet")
    ap.add_argument("--exclude-ids", default="results/cal/clean_2048_v2_ids.txt")
    ap.add_argument("--out-dir", default="ibis_data/raw/tahoe-meta-data/stubgen_sft")
    ap.add_argument("--candidates-per-episode", type=int, default=6)
    ap.add_argument("--max-revealed", type=int, default=16)
    ap.add_argument(
        "--episode-set",
        choices=["train", "holdout"],
        default="train",
        help="train: split==train minus held-out ids; holdout: the held-out eval ids (for a leakage-free val set)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude = set(l.strip() for l in open(args.exclude_ids)) if Path(args.exclude_ids).exists() else set()
    df = pd.read_parquet(args.data_path)
    n0 = len(df)
    if args.episode_set == "train":
        df = df[(df["split"] == "train") & (~df["episode_id"].isin(exclude))]
    else:
        df = df[df["episode_id"].isin(exclude)]
    print(f"episodes: {n0} total -> {len(df)} ({args.episode_set} set)")

    rows = []
    for _, row in df.iterrows():
        ev = [
            s
            for s in row["context_pool"]
            if s.get("kind") == "evidence" and (s.get("y") or {}).get("log2FoldChange") is not None
        ]
        if len(ev) < 2:
            continue
        cands = rng.sample(ev, min(args.candidates_per_episode, len(ev)))
        for cand in cands:
            others = [s for s in ev if s is not cand]
            r = rng.randint(0, min(args.max_revealed, len(others)))
            revealed = rng.sample(others, r)
            rows.append(build_example(row, cand, revealed, rng))

    rng.shuffle(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"messages": [r["messages"] for r in rows], "chat_template_kwargs": [r["chat_template_kwargs"] for r in rows]}
    ).to_parquet(out / "train-00000-of-00001.parquet", index=False)
    print(f"wrote {len(rows)} SFT examples -> {out}/train-00000-of-00001.parquet")
    print("sample user prompt:\n", rows[0]["messages"][0]["content"][:600])
    print("sample completion:", rows[0]["messages"][1]["content"])


if __name__ == "__main__":
    main()
