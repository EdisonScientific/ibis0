"""On-disk memoisation for bootstrap confidence intervals.

Bootstrap CIs dominate render time (1,000+ resamples per metric per model, and sMECE ~35x that per draw),
and the figures get revised far more often than the underlying cells change. Caching turns a re-render from
minutes into seconds.

**The only thing that matters here is that a cache hit is indistinguishable from a fresh computation.** A
stale CI is worse than a slow one: it is silently wrong in a figure that looks finished. So the key folds in
everything that can change the answer:

1. **The data** — a content hash of the records themselves, not a path or an mtime. Re-running inference
   changes the records, which changes the hash, which misses.
2. **The parameters** — `n_boot`, `seed`, `cluster_key`, `domain_key`, metric names, and anything else the
   caller passes in `key`.
3. **The statistic's identity** — callers must name it (see the `stat_id` contract below).
4. **The implementation** — a content hash of the source files that do the resampling. Edit the bootstrap
   and every entry it produced is invalidated automatically, with no version constant to remember to bump.
   A cosmetic edit to those files also invalidates, which is the right trade.

**The `stat_id` contract.** The statistic is an arbitrary callable, often a lambda closing over a metric
name or a threshold. `__qualname__` does not distinguish `lambda rs: brier(rs)` from `lambda rs: auroc(rs)`
— both are `<lambda>` — so deriving identity by reflection would collide and serve one metric's CI for
another. Callers therefore pass an explicit `stat_id` string. **No `stat_id` means no caching**: the
computation runs as before rather than being cached under an unsafe key.

Disable with `IBIS_BOOT_CACHE=0`; relocate with `IBIS_BOOT_CACHE_DIR`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

_ENABLED = os.environ.get("IBIS_BOOT_CACHE", "1") not in ("0", "false", "no")


def cache_dir() -> Path:
    """Cache root. Lives beside the eval data (not in the repo) — it is derived, regenerable state."""
    if d := os.environ.get("IBIS_BOOT_CACHE_DIR"):
        return Path(d)
    return Path(os.environ.get("IBIS_DATA", "ibis_data")) / "eval" / "_cache" / "bootstrap"


def _blake(s: bytes) -> str:
    return hashlib.blake2b(s, digest_size=16).hexdigest()


def fingerprint_records(records: Any) -> str:
    """Content hash of the records a bootstrap will resample.

    Canonical JSON with sorted keys, so dict ordering cannot change the hash. `default=str` keeps it
    total: an unexpected type degrades to its repr rather than raising, which would take down a render.
    """
    return _blake(json.dumps(records, sort_keys=True, default=str, separators=(",", ":")).encode())


def fingerprint_files(paths: Iterable[str | Path]) -> str:
    """Content hash of the implementation files whose behaviour the cached value depends on."""
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(str(x) for x in paths):
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(f"<missing:{p}>".encode())  # a missing impl file must not silently share a key
    return h.hexdigest()


def cached(
    namespace: str,
    key: dict,
    compute: Callable[[], Any],
    impl_files: Iterable[str | Path] = (),
) -> Any:
    """Return `compute()`, memoised on disk under a key derived from `namespace`, `key` and `impl_files`.

    `key` must already contain a data fingerprint (see `fingerprint_records`) — this function cannot know
    which of its arguments were derived from the data.
    """
    if not _ENABLED:
        return compute()
    payload = {"ns": namespace, "key": key, "impl": fingerprint_files(impl_files)}
    digest = _blake(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode())
    path = cache_dir() / namespace / f"{digest}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())["value"]
        except (OSError, ValueError, KeyError):
            pass  # corrupt or truncated entry: recompute and overwrite
    value = compute()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a concurrent reader never observes a partial entry.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"key": payload, "value": value}, default=str))
        tmp.replace(path)
    except OSError:
        pass  # an unwritable cache must never fail a render
    return value
