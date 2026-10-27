"""Arguments and setup shared by the master-table and grouped-metrics renderers.

Both read the same cells, build the same rows and honour the same bootstrap settings, so the plumbing
lives here rather than being duplicated and allowed to drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LIB = next(p for p in Path(__file__).resolve().parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_paths  # noqa: E402
from macro_metrics import build_rows, constant_reference  # noqa: E402
from roster_palette import ORDER, ROSTER  # noqa: E402


def _common_args(ap) -> None:
    """The roster/tree/bootstrap arguments both renderers need, defined once.

    They must agree: both call `build_rows`, so a differing `--n-boot` is not merely slower, it is a
    different `boot_cache` key — the table and the figure would compute separate bootstraps and report
    different CI widths for the same metric. `render_all.sh` passes one value to both for the same reason.
    """
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/primary/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None, help="override the eval tree root")
    ap.add_argument("--keys", default=None, help="comma list of roster keys (default: full roster order)")
    ap.add_argument("--no-constant", action="store_true", help="omit the constant-p reference row and rule")
    ap.add_argument("--allow-partial", action="store_true", help="macro-average over available domains only")
    # Required, NOT defaulted. Every draw count for every renderer is pinned in one place --
    # `render_all.sh` -- because `boot_cache` keys on it and a silently-wrong value yields CI widths that
    # disagree with the paper while rendering perfectly. A default here would be a second copy of that
    # number, free to drift from it. Omitting the flag
    # is an argparse error rather than a plausible wrong figure.
    ap.add_argument("--n-boot", type=int, required=True)
    ap.add_argument(
        "--smece-boot",
        type=int,
        required=True,
        help="bootstrap draws for sMECE only; it is ~35x costlier per draw than the other metrics. "
        "0 reports sMECE as a point estimate with no interval.",
    )


def resolve(args):
    """(slug, root, keys) from the common arguments, failing loudly on an unknown roster key."""
    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else list(ORDER)
    if unknown := [k for k in keys if k not in ROSTER]:
        raise SystemExit(f"unknown roster keys: {unknown} (known: {list(ORDER)})")
    return slug, root, keys


def rows_and_reference(args, slug, root, keys, *, recalibrate: bool):
    """`build_rows` + the constant-p reference, with the shared no-results error."""
    rows, recal_rows = build_rows(
        slug,
        keys,
        recalibrate=recalibrate,
        allow_partial=args.allow_partial,
        n_boot=args.n_boot,
        smece_boot=args.smece_boot,
        root=root,
    )
    if not rows:
        raise SystemExit(f"no results found under {eval_paths.result_dir(slug, '', root=root).parent}")
    const = None if args.no_constant else constant_reference(slug, keys, args.n_boot, args.smece_boot, root=root)
    return rows, recal_rows, const
