"""Rates of the silent fallbacks recorded by bed_thetaprime.py, per domain and policy.

Each trajectory carries a "fallbacks" dict of call and fallback counts: unparsable chooser reply -> random
candidate (chooser_random), out-of-range chooser index wrapped modulo the pool (chooser_out_of_range),
unparsable belief -> 0.5 (belief_default), unparsable simulation -> candidate scored -1 (sim_unparsed),
request failed after retries (chat_failed).

    uv run python scripts/analysis/rq5_bed/bed_fallback_report.py ibis_data/bed_sim/bed_repro64/*.jsonl
"""

import argparse
import collections
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    args = ap.parse_args()
    totals: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    episodes: collections.Counter = collections.Counter()
    for path in args.results:
        for line in open(path):
            r = json.loads(line)
            key = (r["domain"], r["policy"])
            episodes[key] += 1
            totals[key].update(r.get("fallbacks") or {})

    def rate(c: collections.Counter, num: str, den: str) -> str:
        return f"{c[num]}/{c[den]} ({c[num] / c[den]:.1%})" if c[den] else "-"

    print(
        f"{'domain':7} {'policy':13} {'eps':>4}  {'random choice':>18}  {'out-of-range':>16}  "
        f"{'belief->0.5':>20}  {'sim unparsed':>20}  {'failed req':>10}"
    )
    for (domain, policy), c in sorted(totals.items()):
        print(
            f"{domain:7} {policy:13} {episodes[domain, policy]:>4}  "
            f"{rate(c, 'chooser_random', 'chooser_calls'):>18}  {rate(c, 'chooser_out_of_range', 'chooser_calls'):>16}  "
            f"{rate(c, 'belief_default', 'belief_calls'):>20}  {rate(c, 'sim_unparsed', 'sim_calls'):>20}  "
            f"{c['chat_failed']:>10}"
        )


if __name__ == "__main__":
    main()
