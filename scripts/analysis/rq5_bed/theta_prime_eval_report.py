"""Controllability report for the theta' simulator: does the SAME x yield opposite-valence outcomes?

Joins gen_meta.jsonl (requested valence per synth_id, written by
``validate_simulator.py --synth-out``) with the annotate (theta_prime) realized WoE-for-theta'
of the generated y. Reports per-pole realized WoE and — the decisive number — the PAIRED separation:
for each candidate, is realized WoE(support+) > realized WoE(null-)?
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-meta", default="ibis_data/bed_sim/tp_eval/gen_meta.jsonl")
    ap.add_argument("--woe-root", default="ibis_data/bed_sim/tp_eval_woe")
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    args = ap.parse_args()
    gen = {json.loads(l)["synth_id"]: json.loads(l) for l in open(args.gen_meta)}
    realized = {}
    for dom in args.domains:
        p = Path(args.woe_root) / f"{dom}.train.jsonl"
        if not p.exists():
            continue
        for l in open(p):
            d = json.loads(l)
            realized[d["episode_id"]] = float(d["woe_decibans"])

    # emitted-support echo + per-pole realized
    for want in [True, False]:
        sub = [(s, g) for s, g in gen.items() if g["valence"] == want and s in realized]
        echo = np.mean([(g["emitted_support"] == "yes") == want for _, g in sub]) if sub else float("nan")
        rz = np.array([realized[s] for s, _ in sub])
        pole = "SUPPORT+" if want else "NULL-"
        print(f"=== requested {pole} (n={len(sub)}) ===")
        print(f"  emitted_support echoes request: {100 * echo:.0f}%")
        print(
            f"  realized WoE-for-theta': mean={rz.mean():+.2f} p25={np.percentile(rz, 25):+.2f} "
            f"p50={np.percentile(rz, 50):+.2f} p75={np.percentile(rz, 75):+.2f}"
        )
        print(f"  realized correct side of 0: {100 * np.mean(rz > 0 if want else rz < 0):.0f}%\n")

    # PAIRED: same candidate, both poles
    by_cand = defaultdict(dict)
    for s, g in gen.items():
        if s not in realized:
            continue
        base = s.rsplit("__", 1)[0]
        by_cand[base]["P" if g["valence"] else "N"] = realized[s]
    pairs = [(v["P"], v["N"]) for v in by_cand.values() if "P" in v and "N" in v]
    if pairs:
        P = np.array([p for p, _ in pairs])
        N = np.array([n for _, n in pairs])
        print(f"=== PAIRED separation (n={len(pairs)} candidates with both poles) ===")
        print(f"  realized WoE(support+) - WoE(null-): mean={(P - N).mean():+.2f} med={np.median(P - N):+.2f}")
        print(f"  %% candidates with WoE(support+) > WoE(null-): {100 * np.mean(P > N):.0f}%   <- decisive")
        print(f"  %% candidates fully split (support+ >0 AND null- <0): {100 * np.mean((P > 0) & (N < 0)):.0f}%")


if __name__ == "__main__":
    main()
