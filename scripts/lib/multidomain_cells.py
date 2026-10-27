"""Reading the multi-domain eval tree: which cell files a tag selects, and what is in them.

It lives in `lib/` beside `eval_paths.py` because it is the shared reader every renderer builds on.

Nothing here computes a metric or draws anything, which is the point: `macro_metrics.py` keys its
bootstrap cache on its OWN file hash, so cell-reading changes and plotting changes can no longer
invalidate an expensive computation they cannot affect.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import eval_paths

DOMAINS = ("tahoe", "depmap", "gwas")


def cell_paths_for_tag(testset_slug: str, eval_key: str, domain: str, tag: str, root: Path | None = None):
    """The cell files one `tag` selects for a (model, domain), in ascending-k order.

    `tag` is a concrete tag (`primary`, `ctx7`) or the literal `ctx*`, which selects **every** context
    cell — the dense `sensitivity` sweep pooled over deliberation.

    `ctx*` cannot be a plain glob-and-sort: lexical order puts `ctx10` before `ctx2`, and a caller that
    zips these paths against a k axis would silently mislabel every curve. Sorting on the parsed integer
    is the whole reason this helper exists rather than an inline `Path.glob`.
    """
    if tag != "ctx*":
        p = eval_paths.cell_path(testset_slug, eval_key, domain, tag, root=root)
        return [p] if p.exists() else []
    d = eval_paths.result_dir(testset_slug, eval_key, root=root)
    found = []
    for p in d.glob(f"{domain}_ctx*.json"):
        m = re.fullmatch(rf"{re.escape(domain)}_ctx(\d+)", p.stem)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def load_cells(
    testset_slug: str, eval_key: str, root: Path | None = None, tag: str = "primary"
) -> dict[str, list[dict]]:
    """{domain: [record, ...]} for one model, keeping only parsed episodes.

    Unparsed episodes are excluded from the metrics rather than scored as failures — so the count of what
    was dropped must be reported alongside (see `load_attempts` / the Parse fail column), otherwise a row
    that silently lost its hardest episodes looks identical to one that answered all of them.

    `tag` selects which cells to read and defaults to `primary`. It matters that this is explicit: the
    `primary` and `sensitivity` configs share a `<testset>` slug (the slug keys on repo + revision, not
    config) and therefore share a directory, so pointing a renderer's `--testset` at `sensitivity/` does
    NOT by itself change which files are read. Without a tag argument such a renderer silently re-reads
    the `primary` cells and reports them as the sensitivity result.
    """
    out: dict[str, list[dict]] = {}
    for d in DOMAINS:
        kept = []
        for path in cell_paths_for_tag(testset_slug, eval_key, d, tag, root=root):
            rows = json.loads(path.read_text())["results"]
            kept += [r for r in rows if r.get("parsed_prob") is not None]
        if kept:
            out[d] = kept
    return out


def load_attempts(
    testset_slug: str, eval_key: str, root: Path | None = None, tag: str = "primary"
) -> dict[str, tuple[int, int]]:
    """{domain: (n_attempted, n_parsed)}.

    `n_attempted` is every episode the eval tried, including the ones whose probability could not be
    parsed (truncated mid-sentence, refused, or an API error). The eval scripts keep those rows with
    `parsed_prob: null`, so the denominator is recoverable from the cell file itself.
    """
    out: dict[str, tuple[int, int]] = {}
    for d in DOMAINS:
        n_att = n_ok = 0
        seen = False
        for path in cell_paths_for_tag(testset_slug, eval_key, d, tag, root=root):
            rows = json.loads(path.read_text())["results"]
            seen = True
            n_att += len(rows)
            n_ok += sum(r.get("parsed_prob") is not None for r in rows)
        if seen:
            out[d] = (n_att, n_ok)
    return out
