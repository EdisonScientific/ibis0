"""Shared reader for the published multi-domain TEST set.

`testset_passes` was duplicated verbatim in the classification, generative, and frontier evals. That is
the one function in the eval stack that MUST agree across model families: it decides which episodes make
up a pass and how they are tagged, so a divergence would mean two roster rows silently scoring different
episode sets while reporting the same column name. One definition here, imported by all three.
"""


def testset_passes(df) -> list[tuple[str, list[dict]]]:
    """(tag, [record, ...]) passes from a published test-set config.

    Every row of a published config is already one forward pass — `context_pool` pre-truncated, `k` and
    `n_stubs` resolved — so there is no join, no sampling, and no slicing here. The `testset` column says
    how rows group: a `primary` config is a single pass; a `sensitivity` config is one pass per `k`
    (tagged `ctx{k}`) for the dense nested sweep.

    A config mixing both kinds is an error rather than a best-effort split: it means the set was built or
    filtered wrongly, and guessing would produce a plausible-looking table over the wrong episodes.
    """
    kinds = set(df["testset"])
    if kinds == {"primary"}:
        return [("primary", df.to_dict("records"))]
    if kinds == {"sensitivity"}:
        return [(f"ctx{int(k)}", g.to_dict("records")) for k, g in df.groupby("k", sort=True)]
    raise ValueError(f"expected one testset kind (primary|sensitivity) per config, got {sorted(kinds)}")
