"""Build per-category exemplar y pools for the exemplar-EIG baseline.

For each domain, group REAL training-stub outcomes y by their own theta'-WoE sign (from
woe_thetaprime_full): a supporting pool (WoE > +thr) and a refuting pool (WoE < -thr), keyed by
schema category. The exemplar-EIG policy then draws a real supporting y+ and refuting y- of the
candidate's category (instead of the trained simulator's y+/y-), grafts them onto the candidate x,
and runs the same p'-weighted EIG. Isolates the value of the trained simulator vs. real exemplars.

Writes <out>.json = {domain: {category: {"support": [y,...], "refute": [y,...]}}}.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "envs" / "ibis_env_utils").is_dir())
sys.path.insert(0, str(_REPO / "scripts" / "data_pipeline"))
for _m in ("ibis_env_utils", "gwas_meta", "depmap_meta", "tahoe_meta"):
    sys.path.insert(0, str(_REPO / "envs" / _m))
import build_bed_simulator_sft as B
from ibis_env_utils import Split, load_dataset_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--woe-root", default="ibis_data/bed_sim/woe_thetaprime_full")
    ap.add_argument("--data-root", default="ibis_data/bed_sim/data")
    ap.add_argument("--out", default="ibis_data/bed_sim/exemplar_pools.json")
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    ap.add_argument("--cap", type=int, default=400, help="max exemplars per (category, sign)")
    ap.add_argument("--thr", type=float, default=1.0, help="|WoE| threshold to count as a clear exemplar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pools = {}
    for domain in args.domains:
        df = load_dataset_split(f"{args.data_root}/{domain}", Split.TRAIN)
        rows = {r["episode_id"]: r for r in df.to_dict("records")}
        items = []
        for l in open(Path(args.woe_root) / f"{domain}.train.jsonl"):
            d = json.loads(l)
            items.append((d["episode_id"], d["stub_index"], float(d["woe_decibans"])))
        rng.shuffle(items)  # random sample per (cat,sign) rather than first-N
        cat = defaultdict(lambda: {"support": [], "refute": []})
        for eid, i, woe in items:
            r = rows.get(eid)
            if r is None:
                continue
            cp = r.get("context_pool")
            if cp is None or i >= len(cp):
                continue
            stub = cp[i]
            k = B.schema_key(domain, stub)
            if woe > args.thr and len(cat[k]["support"]) < args.cap:
                cat[k]["support"].append(B.y_completion(domain, stub)[0])
            elif woe < -args.thr and len(cat[k]["refute"]) < args.cap:
                cat[k]["refute"].append(B.y_completion(domain, stub)[0])
        pools[domain] = {k: v for k, v in cat.items()}
        for k, v in cat.items():
            print(f"[{domain}] category {k}: support={len(v['support'])} refute={len(v['refute'])}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pools, open(args.out, "w"))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
