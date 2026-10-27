"""Build the conflict and position probe sets, per domain, in the published test-set layout.

Domain knowledge lives in `probe_stubs.py`; this file owns only the pool construction.

Output matches `load_dataset_split`'s config layout so the existing evals consume it unchanged:

    <out-dir>/conflict/<domain>/test.parquet     5 rows per base episode (the conditions below)
    <out-dir>/position/<domain>/test.parquet     4 rows per base episode (probe at each position)

Every row is one forward pass with `context_pool` already final, and carries `domain`, `testset`,
`k` and `n_stubs` like any published config. `testset` is `"primary"` so `testset_passes` yields a
single pass per domain; the condition is encoded in `episode_id` as `<base>__<cond>`, which is what the
analyses split on.

**Domain coverage differs by probe, for reasons in `probe_stubs`.** Conflict runs on Tahoe and DepMap:
GWAS evidence has no refuting category to mirror into. Position runs on all three, at a pool of ten
rather than sixteen, because DepMap episodes carry only ~11 stubs that can
serve as padding and a pool of sixteen leaves 14 eligible episodes out of 2,000.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_SCRIPTS = next(p for p in _HERE.parents if p.name == "scripts")
sys.path.insert(0, str(_SCRIPTS / "lib"))
sys.path.insert(0, str(_HERE))

from ibis_env_utils import Split, load_dataset_split  # noqa: E402
from probe_stubs import MIRRORABLE, is_padding, is_supporting, mirror, support_rank  # noqa: E402

CONFLICT_CONDS = ["none", "support", "refute", "both_sfirst", "both_rfirst"]


def _sha_order(episode_ids):
    """Same seed-free ordering `build_static_testset` uses, so the probe subset is reproducible from
    episode_id alone and does not drift when the selection code is re-run."""
    return sorted(episode_ids, key=lambda e: hashlib.sha256(str(e).encode()).hexdigest())


def conflict_pools(domain, probe):
    """The five conditions. `none` is the prior-only baseline every effect is measured against.

    Both orderings of the conflicting pair are emitted so the analysis can average them, which removes
    order from the additivity residual rather than letting it leak in.
    """
    m = mirror(domain, probe)
    return {
        "none": [],
        "support": [probe],
        "refute": [m],
        "both_sfirst": [probe, m],
        "both_rfirst": [m, probe],
    }


def position_pools(probe, pad, positions, pool_size):
    """The same probe stub at each position, with the same padding in the same order around it.

    Nested by construction: the context preceding the probe at position p is the first p-1 padding
    stubs, identical across conditions, so the probe's weight of evidence at p is comparable to p'.
    """
    out = {}
    for p in positions:
        pool = pad[: p - 1] + [probe] + pad[p - 1 :]
        out[f"pos{p}"] = pool[:pool_size]
    return out


def build_domain(domain, probe_kind, data_path, n_episodes, positions, pool_size, revision=None):
    df = load_dataset_split(data_path, Split.TEST, revision=revision)
    by_id = {str(r["episode_id"]): r for r in df.to_dict("records")}

    rows, kept = [], 0
    skip = {"no_support": 0, "not_enough_padding": 0}
    need_pad = pool_size - 1
    for eid in _sha_order(by_id):
        row = by_id[eid]
        pool = list(row["context_pool"])
        sup = [s for s in pool if is_supporting(domain, s, row)]
        if not sup:
            skip["no_support"] += 1
            continue
        probe = sorted(sup, key=lambda s: support_rank(domain, s))[0]

        if probe_kind == "conflict":
            pools = conflict_pools(domain, probe)
        else:
            pad = [s for s in pool if is_padding(domain, s) and s is not probe]
            if len(pad) < need_pad:
                skip["not_enough_padding"] += 1
                continue
            pools = position_pools(probe, pad[:need_pad], positions, pool_size)

        for cond, cpool in pools.items():
            r = {k: v for k, v in row.items() if k != "context_pool"}
            r["context_pool"] = cpool
            r["episode_id"] = f"{eid}__{cond}"
            r["domain"] = domain
            r["testset"] = "primary"
            r["k"] = len(cpool)
            r["n_stubs"] = len(cpool)
            rows.append(r)
        kept += 1
        if kept >= n_episodes:
            break

    frame = pd.DataFrame(rows)
    print(f"  {domain}: {kept} base episodes -> {len(frame)} rows   skipped: {skip}")
    return frame, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=["conflict", "position"])
    ap.add_argument("--domains", default="", help="comma-separated; default is the probe's own coverage")
    ap.add_argument(
        "--source",
        default="hf://EdisonScientific/EDB-Bio/{domain}",
        help="upstream dataset, which carries UNTRUNCATED context_pool. The published test set will not "
        "do here: its pools are already cut to each episode's own k, so there would be too few stubs "
        "left to build a probe context from.",
    )
    ap.add_argument("--revision", default="v1.0.0", help="revision of --source (ignored for a local path)")
    ap.add_argument("--n-episodes", type=int, default=256, help="base episodes per domain")
    ap.add_argument("--pool-size", type=int, default=10, help="position probe only")
    ap.add_argument("--positions", default="1,4,7,10", help="position probe only; 1-indexed")
    ap.add_argument("--out-dir", default="outputs/probe_sets")
    args = ap.parse_args()

    default_domains = MIRRORABLE if args.probe == "conflict" else ("tahoe", "depmap", "gwas")
    domains = [d for d in args.domains.split(",") if d] or list(default_domains)
    if args.probe == "conflict":
        bad = [d for d in domains if d not in MIRRORABLE]
        if bad:
            raise SystemExit(
                f"the conflict probe has no mirror for {bad}: that evidence has no refuting category, so a "
                f"mirrored stub would be a fabricated record. See probe_stubs.py."
            )
    positions = [int(x) for x in args.positions.split(",")]
    if args.probe == "position" and max(positions) > args.pool_size:
        raise SystemExit(f"position {max(positions)} exceeds pool size {args.pool_size}")

    out = Path(args.out_dir)
    print(f"{args.probe} probe over {domains}")
    total = 0
    for d in domains:
        frame, kept = build_domain(
            d, args.probe, args.source.format(domain=d), args.n_episodes, positions, args.pool_size, args.revision
        )
        if kept < args.n_episodes:
            print(f"  !! {d}: only {kept} of {args.n_episodes} requested episodes were eligible")
        cfg = out / args.probe / d
        cfg.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cfg / "test.parquet", index=False)
        total += len(frame)
        print(f"  -> {cfg / 'test.parquet'}")
    print(f"{total} rows total")


if __name__ == "__main__":
    main()
