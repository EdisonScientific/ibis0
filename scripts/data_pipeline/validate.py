"""Schema validation for evidence-deliberation episode datasets.

Every collection, whatever its biological domain, emits episodes sharing the same
required fields: a target hypothesis with a binary ``supported`` label, and an
ordered pool of evidence stubs each tagged with a relevance tier. Stub *payloads*
are domain-specific, so these checks operate only on those shared fields — they
transfer across all collections. Domain-specific invariants (e.g. "the target's own
measurement never appears in its pool") are passed in per dataset via ``leakage_check``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("episode_id", "context_pool")
SPLITS = ("train", "val", "test")


def _pooled_labels(df: pd.DataFrame, label_col: str) -> list[int]:
    """Flatten labels whether stored as a single int (``label``) or a per-target dict (``labels``)."""
    out: list[int] = []
    if label_col not in df.columns:
        return out
    for v in df[label_col]:
        if isinstance(v, dict):
            out.extend(int(x) for x in v.values())
        elif v is not None:
            out.append(int(v))
    return out


def validate_episodes(
    df: pd.DataFrame,
    *,
    tier_field: str = "context_type",
    label_col: str = "label",
    min_pool: int = 1,
    max_pool: int = 64,
    balance_tol: float = 0.15,
    leakage_check: Callable[[pd.DataFrame], int] | None = None,
    strict: bool = True,
) -> dict:
    """Validate the shared episode fields. Returns a report; raises if ``strict``.

    ``leakage_check`` is an optional per-dataset callable returning the number of
    episodes that leak the held-out target measurement into their evidence pool.
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict = {"n_episodes": len(df)}

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        return _finish(errors, warnings, stats, strict)

    n_dup = int(df["episode_id"].duplicated().sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate episode_id")

    labels = _pooled_labels(df, label_col)
    if not labels:
        errors.append(f"no labels found (looked for column '{label_col}')")
    else:
        rate = sum(labels) / len(labels)
        stats["n_targets"] = len(labels)
        stats["supported_rate"] = round(rate, 4)
        if abs(rate - 0.5) > balance_tol:
            errors.append(f"label imbalance: supported_rate={rate:.3f} (tol {balance_tol})")

    sizes: list[int] = []
    missing_tier = 0
    for pool in df["context_pool"]:
        pool = list(pool) if pool is not None else []
        sizes.append(len(pool))
        missing_tier += sum(1 for s in pool if not dict(s).get(tier_field))
    if sizes:
        stats["pool_size_min"], stats["pool_size_max"] = min(sizes), max(sizes)
        if min(sizes) < min_pool:
            errors.append(f"pool smaller than {min_pool} (min {min(sizes)})")
        if max(sizes) > max_pool:
            warnings.append(f"pool larger than {max_pool} (max {max(sizes)})")
    if missing_tier:
        errors.append(f"{missing_tier} stubs missing tier field '{tier_field}'")

    if leakage_check is not None:
        leaks = int(leakage_check(df))
        stats["leaks"] = leaks
        if leaks:
            errors.append(f"{leaks} episodes leak the target's own measurement into the pool")

    return _finish(errors, warnings, stats, strict)


def _finish(errors, warnings, stats, strict) -> dict:
    report = {"ok": not errors, "errors": errors, "warnings": warnings, "stats": stats}
    if strict and errors:
        raise ValueError("episode validation failed:\n  " + "\n  ".join(errors) + f"\n  stats={stats}")
    return report


def _validate_collection(d: Path, **kw) -> bool:
    """Validate a collection's split files; print a per-split report.

    Also checks that no ``episode_id`` appears in more than one split (a leak across
    train/val/test), which the per-split checks cannot catch on their own.
    """
    ok = True
    ids: dict[str, set] = {}
    for split in SPLITS:
        pq = d / f"{split}.parquet"
        if not pq.exists():
            print(f"[{d.name}/{split}] MISSING {pq}")
            ok = False
            continue
        df = pd.read_parquet(pq)
        ids[split] = set(df["episode_id"]) if "episode_id" in df.columns else set()
        report = validate_episodes(df, strict=False, **kw)
        print(f"[{d.name}/{split}] {'ok' if report['ok'] else 'FAIL'}  {report['stats']}")
        for e in report["errors"]:
            print(f"    error: {e}")
        for w in report["warnings"]:
            print(f"    warn:  {w}")
        ok = ok and report["ok"]
    total = sum(len(s) for s in ids.values())
    union = len(set().union(*ids.values())) if ids else 0
    if total != union:
        print(f"[{d.name}] LEAK: {total - union} episode_id(s) appear in more than one split")
        ok = False
    return ok


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Validate evidence-deliberation collections (each a dir with train.parquet + test.parquet)."
    )
    ap.add_argument("collections", nargs="+", help="collection directories to validate")
    ap.add_argument("--tier-field", default="context_type")
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()
    results = [
        _validate_collection(Path(c), tier_field=args.tier_field, label_col=args.label_col) for c in args.collections
    ]
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
