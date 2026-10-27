"""Render the BED selection-policy figure: x = # selected stubs, y = mean P(ground-truth | revealed),
averaged over test episodes, one line per policy (with SEM band). One panel per domain."""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {
    "theta_2pole": "EIG-simulate",
    "exemplar_eig": "EIG-exemplar",
    "naive_opus5": "naive Opus 5",
    "naive_gpt56": "naive GPT-5.6-sol",
    "naive_base": "naive Base LM",
    "naive_rld": "naive RL'd LM",
    "random": "random",
    "oracle": "Oracle (peeks at y)",
    "mc1": "MC-1",
    "mc16": "MC-16",
}
COLOR = {
    "theta_2pole": "#08519c",
    "exemplar_eig": "#4292c6",
    "naive_opus5": "#374752",
    "naive_gpt56": "#63727f",
    "naive_base": "#8695a2",
    "naive_rld": "#aeb9c3",
    "random": "#c6cfd6",
    "oracle": "#111111",
    "mc1": "#ff9d4d",
    "mc16": "#fbbc04",
}
ORDER = [
    "theta_2pole",
    "exemplar_eig",
    "naive_opus5",
    "naive_gpt56",
    "naive_base",
    "naive_rld",
    "random",
    "mc1",
    "mc16",
    "oracle",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["ibis_data/bed_sim/bed_tp/full_*.jsonl"])
    ap.add_argument("--out", default="ibis_data/bed_sim/bed_tp/bed_figure.png")
    ap.add_argument("--title", default="BED: belief in ground truth vs. #experiments selected")
    ap.add_argument(
        "--min-frac", type=float, default=0.5, help="only plot steps supported by >= this fraction of episodes"
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="single panel pooling all domains (equal weight, capped at the shortest domain's budget)",
    )
    args = ap.parse_args()
    rows = [json.loads(l) for pat in args.results for f in glob.glob(pat) for l in open(f)]
    domains = sorted({r["domain"] for r in rows})

    if args.aggregate:
        # cap at the SHORTEST domain budget so every plotted step has all domains (equal per-domain weight)
        dmax = {}
        for r in rows:
            t = max(pt["t"] for pt in r["traj"])
            dmax[r["domain"]] = max(dmax.get(r["domain"], 0), t)
        cap = min(dmax.values())
        pols = [p for p in ORDER if p in {r["policy"] for r in rows}]
        fig, ax = plt.subplots(figsize=(8.0, 5.8))
        for pol in pols:
            byt = defaultdict(list)
            for r in rows:
                if r["policy"] != pol:
                    continue
                lab = r["label"]
                for pt in r["traj"]:
                    if pt["t"] <= cap:
                        byt[pt["t"]].append(pt["p"] if lab == 1 else 1 - pt["p"])
            ts = sorted(byt)
            mean = np.array([np.mean(byt[t]) for t in ts])
            sem = np.array([np.std(byt[t]) / max(1, np.sqrt(len(byt[t]))) for t in ts])
            ax.plot(
                ts,
                mean,
                "-o",
                ms=4,
                lw=2.4,
                color=COLOR.get(pol, "#666"),
                label=LABEL.get(pol, pol),
                zorder=3 if pol in ("theta_2pole", "exemplar_eig") else 2,
            )
            ax.fill_between(ts, mean - sem, mean + sem, color=COLOR.get(pol, "#666"), alpha=0.12, lw=0)
        nep = sum(1 for r in rows if r["policy"] == pols[0])
        ax.set_title(f"Aggregate across {', '.join(sorted(dmax))}  (n={nep} episode-runs/arm, capped at {cap} steps)")
        ax.set_xlabel("# experiments selected")
        ax.set_ylabel("mean P(ground truth | revealed)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="best")
        fig.suptitle(args.title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=140)
        print(f"wrote {args.out}  (aggregate of {len(rows)} trajectories, cap={cap} steps)")
        return
    fig, axes = plt.subplots(1, len(domains), figsize=(6.2 * len(domains), 5.0), squeeze=False)
    for ax, dom in zip(axes[0], domains):
        sub = [r for r in rows if r["domain"] == dom]
        pols = [p for p in ORDER if p in {r["policy"] for r in sub}]
        for pol in pols:
            byt = defaultdict(list)
            for r in sub:
                if r["policy"] != pol:
                    continue
                lab = r["label"]
                for pt in r["traj"]:
                    byt[pt["t"]].append(pt["p"] if lab == 1 else 1 - pt["p"])
            # only plot steps supported by >= min_frac of the max episode count (drops the noisy tail
            # where few episodes have that many selectable candidates)
            nmax = max((len(v) for v in byt.values()), default=0)
            ts = [t for t in sorted(byt) if len(byt[t]) >= args.min_frac * nmax]
            mean = np.array([np.mean(byt[t]) for t in ts])
            sem = np.array([np.std(byt[t]) / max(1, np.sqrt(len(byt[t]))) for t in ts])
            n_ep = len(byt[ts[0]]) if ts else 0
            ax.plot(
                ts,
                mean,
                "-o",
                ms=3,
                lw=2,
                color=COLOR.get(pol, "#666"),
                label=f"{LABEL.get(pol, pol)}",
                zorder=3 if pol == "theta_2pole" else 2,
            )
            ax.fill_between(ts, mean - sem, mean + sem, color=COLOR.get(pol, "#666"), alpha=0.12, lw=0)
        ax.set_title(f"{dom}  (n={n_ep} episodes)")
        ax.set_xlabel("# experiments selected")
        ax.set_ylabel("mean P(ground truth | revealed)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}  ({len(rows)} trajectories, domains={domains})")


if __name__ == "__main__":
    main()
