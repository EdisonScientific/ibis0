"""Coverage + parse-rate check on the eval tree, between the evals and the renderers.

Answers the two questions a renderer cannot answer for you, because both look identical to success from
inside a figure:

1. **Is every cell there?** A `sensitivity` model is complete at 17 k x 3 domains = **51** cells; a
   `primary` model at 3. A renderer given 14 of 17 k just draws a shorter curve, and a roster row missing a
   domain quietly stops being a 3-domain macro average. Missing cells are listed per model, by domain.
2. **What was the parse rate, per (domain, k)?** Unparsed episodes are excluded from metrics rather than
   scored as wrong, which is the right choice but means a row that lost its hardest episodes looks
   identical to one that answered everything. If the parse rate falls with k, a dose-response curve
   computed on survivors is confounded by the very variable on its x axis — so the rate is reported
   against k, not as one number per model.

Also reconciles `run.json`'s `cells_written` against the directory when `--fix-run-json` is passed. The
launchers write that field before knowing how a config split into passes, so it can name fewer cells than
landed; the directory is the ground truth and every cell carries its own resolved `testset`.

    uv run --no-sync python scripts/analysis/multidomain/verify_cells.py --tag 'ctx*'
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in _HERE.parents if p.name == "scripts") / "lib"))
sys.path.insert(0, str(_HERE))

import eval_paths  # noqa: E402
from multidomain_cells import DOMAINS  # noqa: E402
from roster_palette import ORDER, ROSTER  # noqa: E402

# Leave-one-out classifiers: not roster rows, but the transfer matrix reads them, so a coverage check that
# skipped them would report "complete" while the matrix was missing three of its four rows.
LOO: dict[str, str] = {
    "gwas-depmap__qwen3-30b-a3b__cls__r2.1.0__step_1700": "leave-out-tahoe",
    "gwas-tahoe__qwen3-30b-a3b__cls__r2.1.0__step_1300": "leave-out-depmap",
    "depmap-tahoe__qwen3-30b-a3b__cls__r2.1.0__step_1800": "leave-out-gwas",
}


def scan(slug: str, key: str, root: Path | None) -> tuple[dict, dict]:
    """({(domain, tag): (n_attempted, n_parsed)}, {tag_kind: count}) for one model directory."""
    d = eval_paths.result_dir(slug, key, root=root)
    cells: dict[tuple[str, str], tuple[int, int]] = {}
    if not d.is_dir():
        return cells, {}
    for p in sorted(d.glob("*.json")):
        if p.name == "run.json":
            continue
        stem = p.stem
        dom = next((x for x in DOMAINS if stem.startswith(f"{x}_")), None)
        if dom is None:
            continue
        tag = stem[len(dom) + 1 :]
        try:
            rows = json.loads(p.read_text())["results"]
        except Exception as e:  # a truncated cell is a real failure mode worth naming, not skipping
            print(f"  !! unreadable cell {p.name}: {e}", file=sys.stderr)
            continue
        cells[(dom, tag)] = (len(rows), sum(r.get("parsed_prob") is not None for r in rows))
    kinds: dict[str, int] = defaultdict(int)
    for _, tag in cells:
        kinds["primary" if tag == "primary" else "ctx"] += 1
    return cells, dict(kinds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="hf://EdisonScientific/EDB-Bio/testsets/sensitivity/{domain}")
    ap.add_argument("--revision", default=eval_paths.DEFAULT_TESTSET_REVISION)
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--tag", default="ctx*", help="'ctx*' (expects 51 cells) or 'primary' (expects 3)")
    ap.add_argument("--k-max", type=int, default=16, help="highest k the sensitivity config defines")
    # BooleanOptionalAction, not store_true: with `default=True` a store_true flag can never be turned
    # off, so `--no-include-loo` would silently not exist.
    ap.add_argument(
        "--include-loo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also check the 3 leave-one-out classifiers the transfer matrix reads (default: on)",
    )
    ap.add_argument("--fix-run-json", action="store_true", help="rewrite cells_written from the directory")
    args = ap.parse_args()

    slug = eval_paths.testset_slug(args.testset, args.revision)
    root = Path(args.eval_root) if args.eval_root else None
    pooled = args.tag == "ctx*"
    want_tags = [f"ctx{k}" for k in range(args.k_max + 1)] if pooled else ["primary"]
    expect = len(want_tags) * len(DOMAINS)

    models = [(k, ROSTER[k].label, ROSTER[k].eval_key) for k in ORDER]
    if args.include_loo:
        models += [(v, f"{v} (LOO)", k) for k, v in LOO.items()]

    print(f"eval tree : {eval_paths.result_dir(slug, '', root=root).parent}")
    print(f"tag       : {args.tag}   expecting {expect} cells/model ({len(want_tags)} tags x {len(DOMAINS)} domains)\n")

    incomplete, parse_issues, summary = [], [], []
    for _key, label, ek in models:
        cells, _ = scan(slug, ek, root)
        have = {(d, t) for (d, t) in cells if t in want_tags}
        missing = [(d, t) for d in DOMAINS for t in want_tags if (d, t) not in have]
        att = sum(cells[c][0] for c in have)
        ok = sum(cells[c][1] for c in have)
        rate = ok / att if att else float("nan")
        flag = "OK " if not missing else "SHORT"
        print(f"{flag} {label:26s} {len(have):3d}/{expect} cells   episodes {ok}/{att} parsed ({rate:.2%})")
        if missing:
            by_dom = defaultdict(list)
            for d, t in missing:
                by_dom[d].append(t)
            for d, ts in by_dom.items():
                print(f"       missing {d}: {len(ts)} -> {', '.join(ts[:20])}{' …' if len(ts) > 20 else ''}")
            incomplete.append((label, len(have)))
        summary.append(
            {
                "model": label,
                "eval_key": ek,
                "cells": len(have),
                "expected": expect,
                "attempted": att,
                "parsed": ok,
                "parse_rate": rate,
                "missing": [f"{d}_{t}" for d, t in missing],
            }
        )
        # Per-(domain, k) parse rate: only interesting where it is not 100%, but the k pattern is the point.
        for d in DOMAINS:
            bad = [(t, cells[(d, t)]) for t in want_tags if (d, t) in cells and cells[(d, t)][1] < cells[(d, t)][0]]
            if bad:
                detail = ", ".join(f"{t}:{p}/{a}" for t, (a, p) in bad)
                print(f"       parse<100% {d}: {detail}")
                parse_issues.append((label, d, bad))

    if args.fix_run_json:
        print()
        for _key, label, ek in models:
            p = eval_paths.result_dir(slug, ek, root=root) / "run.json"
            if not p.exists():
                continue
            cells, _ = scan(slug, ek, root)
            actual = sorted(f"{d}_{t}" for (d, t) in cells)
            rec = json.loads(p.read_text())
            if rec.get("cells_written") != actual:
                rec["cells_written"] = actual
                p.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
                print(f"[fix] {label}: cells_written -> {len(actual)} cells")

    print(f"\n{len(models) - len(incomplete)}/{len(models)} models complete at tag '{args.tag}'.")
    if incomplete:
        print("SHORT: " + ", ".join(f"{lbl} ({n}/{expect})" for lbl, n in incomplete))
    if parse_issues:
        print(
            f"{len(parse_issues)} (model, domain) pairs have at least one k below 100% parse — see above; "
            "a dose-response curve over those k is computed on survivors."
        )
    else:
        print("Every cell parsed 100% of its episodes.")
    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs") / f"cell_coverage_{'ctx' if pooled else 'primary'}.json"
    out.write_text(json.dumps({"tag": args.tag, "expected_per_model": expect, "models": summary}, indent=2))
    print(f"[verify] summary -> {out}")


if __name__ == "__main__":
    main()
