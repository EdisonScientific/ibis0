"""Build the multi-domain static TEST sets (stage 1 of the eval pipeline).

Materializes the **full collated evaluation data** — not pointers into the upstream dataset — so that
every model scores byte-identical episodes at identical context lengths. One row = one forward pass, with `context_pool` already
truncated to that row's `k`.

Why materialize rather than emit an (episode_id, k) manifest: a manifest pins *which* episode and
*how many* stubs, but the stub content is re-fetched from the live upstream repo at inference time.
`context_pool` is a fixed, pre-shuffled list, so if the upstream dataset is ever regenerated with a
different shuffle, the same (episode_id, k) silently resolves to a *different* set of k stubs — no
error, no symptom, and results collected either side of the regeneration would be pooled into one
figure. Shipping the content collapses that two-part dependency into a single addressable artifact.

Output layout matches `load_dataset_split`'s `hf://<org>/<repo>/<config>` -> `<config>/<split>.parquet`
convention, so the same tree is consumable as a local directory or from the Hub:

    <out-dir>/primary/<domain>/test.parquet       one row per episode, at its own k in [k_lo, k_hi]
    <out-dir>/sensitivity/<domain>/test.parquet   one row per (episode, k) over the dense grid 0..K
    <out-dir>/<set>/index.jsonl                   (domain, episode_id, k) selection index, no content
    <out-dir>/provenance.json                     source repo + revision, args, per-config counts

Config per (set, domain) rather than one mixed file: the sweeps import a single `<domain>_meta` env
module per invocation, so a mixed-domain table could not be rendered in one pass anyway — and the
per-domain schemas genuinely differ (tahoe carries no `ground_truth`).

Added columns on every row: `domain`, `testset`, `k` (requested) and `n_stubs` (actual, pool-capped —
a requested k above an episode's pool size yields fewer stubs, and downstream must key off the actual
count). All original dataset columns are preserved so `render_user` / `build_record` work unchanged.

Determinism: per-domain episode selection is a seeded permutation keyed by `(--seed, domain)`; the `k`
assignment and the sensitivity subset are keyed by `sha256(episode_id)`, so selection is reproducible
from `episode_id` alone and independent of run/seed drift.

The published copy is the `testsets/` folder of `EdisonScientific/EDB-Bio` at tag `v1.0.0`; a rebuild
from the `v1.0.0` source at default arguments reproduces it byte for byte.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

_ENVS = next(p for p in Path(__file__).resolve().parents if p.name == "scripts").parent / "envs"
import sys  # noqa: E402

sys.path.insert(0, str(_ENVS / "ibis_env_utils"))
from ibis_env_utils import Split, load_dataset_split  # noqa: E402


def _domain_rng(seed: int, domain: str) -> np.random.Generator:
    """A per-domain generator keyed by (seed, domain) so domains subsample independently."""
    h = hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def _sha_order(episode_ids: list) -> list:
    """Episode ids ordered by sha256 digest — a content-independent, reproducible shuffle."""
    return sorted(episode_ids, key=lambda e: hashlib.sha256(str(e).encode()).hexdigest())


def select_domain(
    domain: str,
    data_path: str,
    primary_n: int,
    k_lo: int,
    k_hi: int,
    sens_n: int,
    seed: int,
    revision: str | None = None,
) -> tuple[list[tuple], list[tuple]]:
    """Return (primary_pairs, sensitivity_pairs) as [(episode_id, k), ...] for one domain.

    Selection only — no episode content is touched here, which keeps this function cheap to test and
    lets the emitted index.jsonl be diffed against any earlier manifest to prove selection is stable.
    """
    df = load_dataset_split(data_path, Split.TEST, columns=["episode_id"], revision=revision)
    eids = df["episode_id"].tolist()
    if any(e is None for e in eids):
        raise ValueError(f"{domain}: TEST split has null episode_id(s); the test set needs stable ids.")
    if len(set(eids)) != len(eids):
        raise ValueError(f"{domain}: TEST split has duplicate episode_ids.")
    if len(eids) < primary_n:
        raise ValueError(f"{domain}: TEST split has {len(eids)} episodes < --primary-n {primary_n}.")

    # Seeded subsample of primary_n, then a content-independent sha order for k assignment.
    rng = _domain_rng(seed, domain)
    picked = [eids[i] for i in rng.permutation(len(eids))[:primary_n]]
    ordered = _sha_order(picked)

    # Balanced k over {k_lo..k_hi}. When n_k does not divide primary_n the surplus cannot be spread
    # evenly, so the r leftover episodes go to the r LARGEST k values, one each. That holds the
    # worst-case imbalance to a single episode and puts the surplus at the longest contexts, where the
    # extra sample is most useful. Uniform random sampling was considered instead and rejected: it is
    # *less* balanced (multinomial sd ~11 per k at primary_n=2000) for no analytic gain, since the
    # primary set is only ever used for metrics that marginalise over k.
    #   primary_n=900  -> r=0, exactly 60 per k
    #   primary_n=2000 -> r=5, 134 for k in {12..16} and 133 for k in {2..11}
    n_k = k_hi - k_lo + 1
    base, r = divmod(primary_n, n_k)
    counts = {k: base + (1 if k > k_hi - r else 0) for k in range(k_lo, k_hi + 1)}
    ks = [k for k in range(k_lo, k_hi + 1) for _ in range(counts[k])]
    primary_pairs = list(zip(ordered, ks, strict=True))

    # Sensitivity subset: first sens_n of the same sha order (nested within primary), dense 0..k_hi.
    sensitivity_pairs = [(e, k) for e in ordered[:sens_n] for k in range(0, k_hi + 1)]
    return primary_pairs, sensitivity_pairs


def materialize(domain: str, data_path: str, pairs: list[tuple], testset: str, revision: str | None = None):
    """Expand [(episode_id, k), ...] into a DataFrame with one row per forward pass.

    `context_pool` is truncated to k on each row; `n_stubs` records the ACTUAL post-cap count, which
    is what every downstream metric and figure must key off (a requested k above an episode's pool
    size yields fewer stubs — silently, by design, since the pool is the hard ceiling).
    """
    df = load_dataset_split(data_path, Split.TEST, revision=revision)
    by_id = {r["episode_id"]: r for r in df.to_dict("records")}
    missing = sorted({e for e, _ in pairs if e not in by_id})
    if missing:
        raise ValueError(f"{domain}: {len(missing)} selected episodes absent from TEST (e.g. {missing[:3]})")

    import pandas as pd

    rows = []
    for eid, k in pairs:
        r = dict(by_id[eid])
        pool = list(r["context_pool"])[:k]
        r["context_pool"] = pool
        r["domain"] = domain
        r["testset"] = testset
        r["k"] = int(k)
        r["n_stubs"] = len(pool)
        rows.append(r)
    return pd.DataFrame(rows)


def _write_provenance(out: Path, payload: dict) -> None:
    """Write provenance.json with sorted keys so a rebuild diffs cleanly."""
    (out / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="gwas,depmap,tahoe")
    ap.add_argument("--data-path-template", default="hf://EdisonScientific/EDB-Bio/{domain}")
    ap.add_argument("--revision", default="v1.0.0", help="source dataset revision (ignored for a local path)")
    ap.add_argument("--out-dir", default="outputs/static_testset")
    ap.add_argument("--primary-n", type=int, default=2000, help="episodes per domain in the primary set")
    ap.add_argument("--primary-k-range", default="2,16", help="lo,hi inclusive; primary k drawn balanced here")
    ap.add_argument("--sensitivity-n", type=int, default=256, help="episodes per domain in the dense sensitivity set")
    ap.add_argument("--sensitivity-k-max", type=int, default=16, help="dense grid is {0..this}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    k_lo, k_hi = (int(x) for x in args.primary_k_range.split(","))
    if not (0 <= k_lo <= k_hi):
        raise ValueError(f"--primary-k-range must be lo<=hi and >=0, got {args.primary_k_range!r}")
    if args.sensitivity_n > args.primary_n:
        raise ValueError("--sensitivity-n must be <= --primary-n (sensitivity is a subset of primary).")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_k = k_hi - k_lo + 1
    print(
        f"building static test sets: domains={domains} primary_n={args.primary_n} k={k_lo}..{k_hi} "
        f"({n_k} values) sensitivity_n={args.sensitivity_n} dense_k=0..{args.sensitivity_k_max} seed={args.seed}"
    )

    index = {"primary": [], "sensitivity": []}
    counts = {}
    for d in domains:
        data_path = args.data_path_template.format(domain=d)
        ppairs, spairs = select_domain(
            d, data_path, args.primary_n, k_lo, k_hi, args.sensitivity_n, args.seed, args.revision
        )
        for testset, pairs in (("primary", ppairs), ("sensitivity", spairs)):
            frame = materialize(d, data_path, pairs, testset, args.revision)
            cfg = out / testset / d
            cfg.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cfg / "test.parquet", index=False)
            counts[f"{testset}/{d}"] = {
                "rows": int(len(frame)),
                "episodes": int(frame["episode_id"].nunique()),
                "bytes": (cfg / "test.parquet").stat().st_size,
            }
            index[testset] += [{"domain": d, "episode_id": e, "k": int(k)} for e, k in pairs]

        hist = {k: 0 for k in range(k_lo, k_hi + 1)}
        for _, k in ppairs:
            hist[k] += 1
        vals = sorted(set(hist.values()))
        print(
            f"  {d:7s}: primary={len(ppairs)} rows (k-counts {vals[0]}..{vals[-1]}, "
            f"{'BALANCED' if len(vals) == 1 else 'near-balanced'})  sensitivity={len(spairs)} rows"
        )

    for testset, rows in index.items():
        (out / testset / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    provenance = {
        "source": args.data_path_template,
        "source_revision": args.revision,
        "builder": "scripts/analysis/multidomain/build_static_testset.py",
        "args": {k: v for k, v in vars(args).items() if k != "out_dir"},
        "configs": counts,
    }
    _write_provenance(out, provenance)
    total = sum(c["rows"] for c in counts.values())
    print(f"wrote {len(counts)} configs under {out} ({total} rows total)")
    for cfg, c in sorted(counts.items()):
        print(f"  {cfg:22s} {c['rows']:5d} rows  {c['episodes']:4d} episodes  {c['bytes'] / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
