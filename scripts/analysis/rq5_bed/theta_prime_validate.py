"""Validate that the belief model scores each stub's OWN hypothesis theta' sensibly.

Joins the theta_prime WoE labels (annotate ANNOT_MODE=theta_prime) with each stub's real outcome y,
and checks that WoE-for-theta' is strongly POSITIVE when the real y is significant (the effect theta'
asserts is present) and ~0/negative when the real y is null. This is the check that justifies using
belief-model theta'-sign labels (rather than domain rules) to build the simulator dataset.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "envs" / "ibis_env_utils").is_dir())
sys.path.insert(0, str(_REPO / "scripts" / "data_pipeline"))
for _m in ("ibis_env_utils", "gwas_meta", "depmap_meta", "tahoe_meta"):
    sys.path.insert(0, str(_REPO / "envs" / _m))
from ibis_env_utils import Split, load_dataset_split


def num(v):
    try:
        return float(v)
    except Exception:
        return None


def significant(domain, y):
    """Domain rule: does the real outcome y indicate the effect theta' asserts is present?"""
    if not isinstance(y, dict):
        return None
    if domain == "gwas":
        p = num(y.get("p_value"))
        return None if p is None else p < 5e-8
    if domain == "depmap":  # crispr dependency
        ch, pd = num(y.get("chronos")), num(y.get("p_dependency"))
        return None if ch is None or pd is None else (ch <= -0.5 and pd >= 0.5)
    padj, lfc = num(y.get("padj")), num(y.get("log2FoldChange"))  # tahoe evidence DE
    return None if padj is None or lfc is None else (padj < 0.05 and abs(lfc) > 0.5)


def strength(domain, y):
    """A continuous significance strength for correlation (higher = stronger effect)."""
    if domain == "gwas":
        p = num(y.get("p_value"))
        return None if not p else -np.log10(max(p, 1e-300))
    if domain == "depmap":
        return None if num(y.get("chronos")) is None else -num(y.get("chronos"))
    lfc = num(y.get("log2FoldChange"))
    return None if lfc is None else abs(lfc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--woe-root", default="ibis_data/bed_sim/woe_thetaprime")
    ap.add_argument("--data-root", default="ibis_data/bed_sim/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    args = ap.parse_args()
    for domain in args.domains:
        p = Path(args.woe_root) / f"{domain}.{args.split}.jsonl"
        if not p.exists():
            print(f"[{domain}] no labels at {p}")
            continue
        df = load_dataset_split(f"{args.data_root}/{domain}", Split(args.split))
        rows = {r["episode_id"]: r for r in df.to_dict("records")}
        woe, sig, strg = [], [], []
        for l in open(p):
            d = json.loads(l)
            r = rows.get(d["episode_id"])
            if r is None or d["stub_index"] >= len(r["context_pool"]):
                continue
            y = (r["context_pool"][d["stub_index"]] or {}).get("y")
            s = significant(domain, y)
            if s is None:
                continue
            woe.append(d["woe_decibans"])
            sig.append(s)
            st = strength(domain, y)
            strg.append(st if st is not None else np.nan)
        woe, sig, strg = np.array(woe), np.array(sig), np.array(strg)
        pos, neg = woe[sig], woe[~sig]
        print(f"\n=== {domain}  (n={len(woe)}, significant={100 * sig.mean():.0f}%) ===")
        print(
            f"  WoE-for-theta' | REAL y SIGNIFICANT: mean={pos.mean():+.2f} med={np.median(pos):+.2f}  %>0={100 * (pos > 0).mean():.0f}%"
        )
        print(
            f"  WoE-for-theta' | REAL y NULL:        mean={neg.mean():+.2f} med={np.median(neg):+.2f}  %>0={100 * (neg > 0).mean():.0f}%"
        )
        m = ~np.isnan(strg)
        if m.sum() > 10 and np.std(strg[m]) > 0 and np.std(woe[m]) > 0:
            print(f"  corr(WoE, effect-strength) = {np.corrcoef(strg[m], woe[m])[0, 1]:+.2f}")
        auc = (pos.reshape(-1, 1) > neg.reshape(1, -1)).mean() if len(pos) and len(neg) else float("nan")
        print(f"  separation AUC (P[WoE_sig > WoE_null]) = {auc:.2f}   -> want >> 0.5")


if __name__ == "__main__":
    main()
