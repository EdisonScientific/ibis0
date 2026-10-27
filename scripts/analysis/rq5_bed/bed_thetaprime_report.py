"""Report for bed_thetaprime: belief-toward-truth convergence per policy + faithfulness-in-loop.

Belief-toward-truth at step t = p if label==1 else 1-p (higher = more correct). We report the mean
trajectory per policy across episodes, steps-to-threshold, and (theta_2pole) whether y+/y- keep
realizing the intended theta' valence as the info set deepens.
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["ibis_data/bed_sim/bed_tp/*.jsonl"])
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()
    files = [f for pat in args.results for f in glob.glob(pat)]
    rows = [json.loads(l) for f in files for l in open(f)]
    print(f"loaded {len(rows)} trajectories from {len(files)} files\n")

    # belief-toward-truth curves per (domain, policy)
    for domain in sorted({r["domain"] for r in rows}):
        print(f"================= {domain} =================")
        pols = sorted({r["policy"] for r in rows if r["domain"] == domain})
        # collect toward-truth per t
        curves = {}
        for pol in pols:
            sub = [r for r in rows if r["domain"] == domain and r["policy"] == pol]
            byt = defaultdict(list)
            for r in sub:
                lab = r["label"]
                for pt in r["traj"]:
                    byt[pt["t"]].append(pt["p"] if lab == 1 else 1 - pt["p"])
            curves[pol] = {t: float(np.mean(v)) for t, v in sorted(byt.items())}
        maxt = max(max(c) for c in curves.values())
        # table: rows = t, cols = policies
        hdr = "  t  | " + " | ".join(f"{p:>12}" for p in pols)
        print(hdr)
        print("  " + "-" * (len(hdr)))
        for t in range(0, maxt + 1):
            cells = " | ".join(f"{curves[p].get(t, float('nan')):>12.3f}" for p in pols)
            print(f"  {t:>2} | {cells}")
        # steps to threshold + AUC (mean toward-truth over steps)
        print(f"\n  {'policy':>12} | steps_to_{args.threshold:.2f} | mean_toward_truth(AUC)")
        for p in pols:
            c = curves[p]
            hit = next((t for t in sorted(c) if c[t] >= args.threshold), None)
            auc = float(np.mean(list(c.values())))
            print(f"  {p:>12} | {str(hit) if hit is not None else '  never':>12} | {auc:.3f}")
        print()

    # faithfulness-in-loop (theta_2pole)
    faith = [
        (f["ctx"], f["yplus_theta_prime"], f["yminus_theta_prime"]) for r in rows if r.get("faith") for f in r["faith"]
    ]
    if faith:
        ctx = np.array([x[0] for x in faith])
        fp = np.array([x[1] for x in faith])
        fm = np.array([x[2] for x in faith])
        print("================= FAITHFULNESS-IN-LOOP (theta_2pole) =================")
        print(f"  n={len(faith)} sim calls logged")
        print(f"  y+ realizes theta'-support: P(theta'|y+) mean={fp.mean():.2f}  %>0.5={100 * (fp > 0.5).mean():.0f}%")
        print(f"  y- realizes theta'-null:    P(theta'|y-) mean={fm.mean():.2f}  %<0.5={100 * (fm < 0.5).mean():.0f}%")
        print(f"  separation (y+ > y-): {100 * np.mean(fp > fm):.0f}%")
        # by context depth bucket
        for lo, hi in [(0, 2), (3, 6), (7, 11), (12, 99)]:
            m = (ctx >= lo) & (ctx <= hi)
            if m.sum():
                print(
                    f"    ctx {lo}-{hi}: n={m.sum()} y+={fp[m].mean():.2f} y-={fm[m].mean():.2f} sep={100 * np.mean(fp[m] > fm[m]):.0f}%"
                )


if __name__ == "__main__":
    main()
