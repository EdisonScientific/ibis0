"""Build the theta' (local-hypothesis) evidence-simulator SFT dataset.

Each candidate experiment x directly tests its OWN hypothesis theta'. We teach the simulator two
coherent, natural conditional generators — a COMPELLING SUPPORTING outcome for theta' and a NULL /
against outcome — so at BED time we can draw both poles for any x and marginalize their influence on
the global target theta by p' = P(theta' | I_t) from the belief model.

No counterfactual synthesis: every example is a REAL (x, y). We just relabel each real stub by the
valence of its own theta' (significant vs null), which the belief model was shown to score cleanly
(theta_prime_validate: separation AUC 0.89-0.98). Context-free: the simulator needs only (theta', x,
requested valence); the info set enters BED only through p'.

Prompt  = (theta' hypothesis, candidate x setup [result hidden], requested valence)
Completion = {"supports_hypothesis": yes|no, "weight_of_evidence_decibans": <belief WoE for theta'>, "y": {...real y...}}

Scope = clean subset only (gwas all / depmap crispr / tahoe evidence) — the woe_thetaprime labels
already contain only those stubs.
"""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "envs" / "ibis_env_utils").is_dir())
sys.path.insert(0, str(_REPO / "scripts" / "data_pipeline"))
for _m in ("ibis_env_utils", "gwas_meta", "depmap_meta", "tahoe_meta"):
    sys.path.insert(0, str(_REPO / "envs" / _m))
import build_bed_simulator_sft as B
from ibis_env_utils import Split, load_dataset_split

THETA_SYSTEM = (
    "You are a simulator of experimental outcomes in biology. You are given the hypothesis that a "
    "candidate experiment directly tests and the experiment's setup, with its result hidden. Produce a "
    "plausible outcome of the REQUESTED valence: either compelling support for the hypothesis, or a "
    "clear null result that argues against it. State whether the outcome supports the hypothesis and "
    "the weight of evidence in decibans (signed: positive supports, negative refutes) it contributes, "
    "then the measured values. Respond with a single JSON object and nothing else."
)


def theta_prime_row(domain, stub):
    """A minimal row whose target_* fields render the stub's OWN direction-agnostic hypothesis theta'."""
    base = {"episode_id": "tp", "label": 0, "ood_group": None}
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
    else:
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


def num(v):
    try:
        return float(v)
    except Exception:
        return None


def significant(domain, y):
    """Domain rule for whether the real outcome supports theta' (the effect is present)."""
    if not isinstance(y, dict):
        return None
    if domain == "gwas":
        p = num(y.get("p_value"))
        return None if p is None else p < 5e-8
    if domain == "depmap":
        ch, pd_ = num(y.get("chronos")), num(y.get("p_dependency"))
        return None if ch is None or pd_ is None else (ch <= -0.5 and pd_ >= 0.5)
    padj, lfc = num(y.get("padj")), num(y.get("log2FoldChange"))
    return None if padj is None or lfc is None else (padj < 0.05 and abs(lfc) > 0.5)


def build_example(env, domain, stub, support, woe, rng):
    """Context-free theta' example: (theta', candidate x setup, requested valence) -> {support, woe, y}."""
    theta = env.render_target(theta_prime_row(domain, stub))
    y, keys = B.y_completion(domain, stub)
    lines = [
        "HYPOTHESIS UNDER TEST (what this experiment directly measures)",
        "=============================================================",
        theta,
        "",
        "CANDIDATE EXPERIMENT (setup only — predict its outcome)",
        "======================================================",
    ]
    lines.extend(env.render_evidence_stub(1, stub, hide_result=True))
    valence = (
        "compelling support for the hypothesis" if support else "a clear null result (evidence against the hypothesis)"
    )
    lines += [
        "",
        "REQUESTED OUTCOME",
        "=================",
        f"valence: {valence}",
        "",
        'Respond with a single JSON object with keys "supports_hypothesis" ("yes"/"no"), '
        '"weight_of_evidence_decibans" (number), and "y" (an object with keys ' + ", ".join(keys) + ").",
    ]
    completion = json.dumps(
        {"supports_hypothesis": "yes" if support else "no", "weight_of_evidence_decibans": round(float(woe), 1), "y": y}
    )
    return {
        "messages": [
            {"role": "system", "content": THETA_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
            {"role": "assistant", "content": completion},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "domain": domain,
        "valence": support,
    }


def build_split(domain, split, args, rng):
    env = importlib.import_module(f"{domain}_meta")
    df = load_dataset_split(f"{args.data_root}/{domain}", Split(split))
    rows = {r["episode_id"]: r for r in df.to_dict("records")}
    pos, neg = [], []
    path = Path(args.woe_root) / f"{domain}.{split}.jsonl"
    for l in open(path):
        d = json.loads(l)
        r = rows.get(d["episode_id"])
        if r is None or d["stub_index"] >= len(r["context_pool"]):
            continue
        stub = r["context_pool"][d["stub_index"]]
        woe = float(d["woe_decibans"])
        if args.pole_source == "rule":
            sig = significant(domain, stub.get("y"))
            if sig is None:
                continue
            support = sig
        else:  # woe sign, with an ambiguous dead-band dropped
            if abs(woe) < args.min_abs_woe:
                continue
            support = woe > 0
        (pos if support else neg).append(build_example(env, domain, stub, support, woe, rng))
    rng.shuffle(pos)
    rng.shuffle(neg)
    if args.balance:
        n = min(len(pos), len(neg))
        pos, neg = pos[:n], neg[:n]
    out = pos + neg
    rng.shuffle(out)
    if args.max_per_domain and len(out) > args.max_per_domain:
        out = out[: args.max_per_domain]
    print(
        f"[{split}] {domain}: {len(out)} examples ({sum(e['valence'] for e in out)} support+ / "
        f"{sum(not e['valence'] for e in out)} null-)",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--woe-root", default="ibis_data/bed_sim/woe_thetaprime_full")
    ap.add_argument("--data-root", default="ibis_data/bed_sim/data")
    ap.add_argument("--out-dir", default="ibis_data/bed_sim/sft_thetaprime")
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument(
        "--pole-source",
        choices=["rule", "woe"],
        default="woe",
        help="valence label: belief WoE sign (coherent with emitted woe; default) or domain rule",
    )
    ap.add_argument("--min-abs-woe", type=float, default=1.0, help="woe pole-source: drop |WoE|<this dead-band")
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--max-per-domain", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rng = random.Random(args.seed)
        rows = []
        for domain in args.domains:
            if not (Path(args.woe_root) / f"{domain}.{split}.jsonl").exists():
                print(f"[{split}] {domain}: no woe labels, skipping")
                continue
            rows.extend(build_split(domain, split, args, rng))
        rng.shuffle(rows)
        if not rows:
            continue
        pd.DataFrame(
            {
                "messages": [r["messages"] for r in rows],
                "chat_template_kwargs": [r["chat_template_kwargs"] for r in rows],
                "domain": [r["domain"] for r in rows],
            }
        ).to_parquet(out / f"{split}.parquet", index=False)
        print(f"[{split}] wrote {len(rows)} -> {out}/{split}.parquet")
        print(f"  sample completion: {rows[0]['messages'][2]['content']}")


if __name__ == "__main__":
    main()
