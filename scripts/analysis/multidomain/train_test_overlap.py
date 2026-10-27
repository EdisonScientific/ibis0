"""Measure train/test overlap in the evidence pools of the published evidence-deliberation dataset.

Episodes are assigned to train/val/test i.i.d., and every pool is assembled before the split exists, so
P5 (a target's own measurement never appears in its own pool) says nothing about OTHER episodes. A
measurement that is the held-out target of a TEST episode can sit inside a TRAIN episode's pool as an
evidence stub, complete with the readout the label is computed from. This script counts how often that
happens, per domain, and writes the episode ids so `render_train_test_overlap.py` can score the roster
with and without them. Paper: Appendix A.7 ("Known limitations of dataset construction").

Two facts about the measurement decide what "the same measurement" means, and they are the only
domain knowledge in this file:

    domain   target key                    stub key (same tuple, read from the stub payload)
    tahoe    (drug, cell line, gene)       drug_metadata.drug, cell_metadata.cell_name, y.gene
    depmap   (gene, cell line id)          gene, cell_metadata.model_id        [crispr stubs only]
    gwas     (SNP, trait)                  snp, trait                          [association stubs]

Only the first `--rendered-k` stubs of a pool count (default 16): that is all the training renderer ever
shows, so a Tahoe pool's positions 16..31 cannot have been seen by any model. Counting the whole stored
pool instead is an upper bound, reported separately.

Also reported, because the spread across domains is otherwise surprising: the size of each domain's
measurement space (the set of possible target keys) and the fraction of it the training pools expose.
Under uniform sampling the overlap rate would equal that coverage; GWAS is not a product space, so its
entry is the set of associations observed anywhere in the dataset and its coverage is relative to that.

Pin `--dataset-revision` to the revision the test set was BUILT from. The eval-tree slug pins the
published TEST set, but the pools live in the source dataset, and reading them from a moving `main`
compares a test set against training pools it was never split against.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = next(p for p in _HERE.parents if p.name == "scripts").parent
sys.path.insert(0, str(_ROOT / "scripts" / "lib"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "envs" / "ibis_env_utils"))

import display_names as dn  # noqa: E402
from multidomain_cells import DOMAINS  # noqa: E402

DEFAULT_DATASET = "hf://EdisonScientific/EDB-Bio/{domain}"
RENDERED_K = 16


# ---------------------------------------------------------------------------------------------------
# Measurement identity
# ---------------------------------------------------------------------------------------------------
def target_key(domain: str, row: dict) -> tuple:
    """The identity of an episode's held-out measurement."""
    if domain == "tahoe":
        return (row["target_drug"], row["target_cell"], row["target_gene"])
    if domain == "depmap":
        return (row["target_gene"], row["target_cell_id"])
    if domain == "gwas":
        return (row["target_snp"], row["target_trait"])
    raise ValueError(f"unknown domain {domain!r}")


def stub_key(domain: str, stub: dict) -> tuple | None:
    """The identity of the measurement an evidence stub reports, or None for stubs that carry no
    target-shaped measurement (aggregates, controls of another kind, other modalities)."""
    if domain == "tahoe":
        y = stub.get("y") or {}
        if stub.get("kind") != "evidence" or y.get("log2FoldChange") is None:
            return None
        k = (
            (stub.get("drug_metadata") or {}).get("drug"),
            (stub.get("cell_metadata") or {}).get("cell_name"),
            y.get("gene"),
        )
    elif domain == "depmap":
        if stub.get("modality") != "crispr":
            return None
        k = (stub.get("gene"), (stub.get("cell_metadata") or {}).get("model_id"))
    elif domain == "gwas":
        if stub.get("modality") != "association":
            return None
        k = (stub.get("snp"), stub.get("trait"))
    else:
        raise ValueError(f"unknown domain {domain!r}")
    return None if any(v is None for v in k) else k


def exposed_measurements(domain: str, episodes: list[dict], k: int | None) -> set[tuple]:
    """Every measurement that appears as a stub within the first `k` positions of any pool (all positions
    when `k` is None)."""
    out: set[tuple] = set()
    for ep in episodes:
        pool = list(ep["context_pool"])
        for stub in pool if k is None else pool[:k]:
            key = stub_key(domain, stub)
            if key is not None:
                out.add(key)
    return out


def measurement_space(domain: str, episodes: list[dict], exposed: set[tuple]) -> tuple[int, str]:
    """(size, how it was counted). Product of the target axes for Tahoe and DepMap; for GWAS the set of
    associations observed anywhere in the dataset, since a (SNP, trait) pair exists only if a study
    reported it."""
    if domain == "gwas":
        universe = exposed | {target_key(domain, ep) for ep in episodes}
        return len(universe), "distinct (SNP, trait) associations observed as a target or stub anywhere in the dataset"
    cols = {"tahoe": ("target_drug", "target_cell", "target_gene"), "depmap": ("target_gene", "target_cell_id")}[domain]
    names = {
        "target_drug": "drugs",
        "target_cell": "cell lines",
        "target_cell_id": "cell lines",
        "target_gene": "genes",
    }
    sizes = [len({ep[c] for ep in episodes}) for c in cols]
    size = 1
    for s in sizes:
        size *= s
    return size, " x ".join(f"{s:,} {names[c]}" for s, c in zip(sizes, cols))


# ---------------------------------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------------------------------
def overlap(domain: str, train: list[dict], val: list[dict], test: list[dict], k: int = RENDERED_K) -> dict:
    """Overlap statistics for one domain. `train`/`val`/`test` are episode dicts with the target columns
    and `context_pool`."""
    fit = train + val
    exposed_k = exposed_measurements(domain, fit, k)
    exposed_all = exposed_measurements(domain, fit, None)
    fit_targets = {target_key(domain, ep) for ep in fit}
    leaked, leaked_any, dup = [], 0, 0
    for ep in test:
        key = target_key(domain, ep)
        if key in exposed_k:
            leaked.append(ep["episode_id"])
        leaked_any += key in exposed_all
        dup += key in fit_targets
    space, space_note = measurement_space(domain, fit + test, exposed_k)
    return {
        "domain": domain,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "rendered_k": k,
        "measurement_space": space,
        "measurement_space_note": space_note,
        "exposed_as_stubs": len(exposed_k),
        "coverage": len(exposed_k) / space if space else float("nan"),
        "test_hypothesis_also_train_target": dup,
        "test_measurement_leaked": len(leaked),
        "test_measurement_leaked_any_position": leaked_any,
        "leaked_episode_ids": sorted(leaked),
    }


def _load_domain(dataset: str, domain: str, revision: str) -> dict[str, list[dict]]:
    from ibis_env_utils import Split, load_dataset_split  # heavy import, kept out of the pure functions

    cols = {
        "tahoe": ["episode_id", "target_drug", "target_cell", "target_gene", "context_pool"],
        "depmap": ["episode_id", "target_gene", "target_cell_id", "context_pool"],
        "gwas": ["episode_id", "target_snp", "target_trait", "context_pool"],
    }[domain]
    path = dataset.format(domain=domain)
    return {s.value: load_dataset_split(path, s, columns=cols, revision=revision).to_dict("records") for s in Split}


def _md_table(rows: list[dict]) -> str:
    head = "| Domain | Measurement space | Exposed as stubs | Coverage | Test hypothesis also a train target | Test measurement leaked |"
    out = [head, "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        n = r["n_test"]
        out.append(
            f"| {dn.domain(r['domain'])} | {r['measurement_space']:,} | {r['exposed_as_stubs']:,} | "
            f"{100 * r['coverage']:.2f}% | {r['test_hypothesis_also_train_target']} / {n:,} "
            f"({100 * r['test_hypothesis_also_train_target'] / n:.1f}%) | {r['test_measurement_leaked']:,} / {n:,} "
            f"({100 * r['test_measurement_leaked'] / n:.1f}%) |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="source dataset, `{domain}` template")
    ap.add_argument(
        "--dataset-revision",
        required=True,
        help="Hub revision of the SOURCE dataset the test set was built from (the paper pins tag v1.0.0)",
    )
    ap.add_argument("--rendered-k", type=int, default=RENDERED_K, help="pool positions a model can see")
    ap.add_argument("--out-dir", default="outputs/train_test_overlap")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DOMAINS:
        print(f"[overlap] {d}: loading train/val/test @ {args.dataset_revision[:8]} …", flush=True)
        splits = _load_domain(args.dataset, d, args.dataset_revision)
        r = overlap(d, splits["train"], splits["val"], splits["test"], k=args.rendered_k)
        rows.append(r)
        print(
            f"[overlap] {d}: {r['test_measurement_leaked']}/{r['n_test']} test measurements appear in a "
            f"train/val pool (first {args.rendered_k}); coverage {100 * r['coverage']:.2f}% of "
            f"{r['measurement_space']:,}"
        )

    ids = {r["domain"]: {"leaked": r["leaked_episode_ids"], "n_test": r["n_test"]} for r in rows}
    (out / "overlap_ids.json").write_text(
        json.dumps({"dataset_revision": args.dataset_revision, "rendered_k": args.rendered_k, "domains": ids}, indent=1)
    )
    table_rows = [{k: v for k, v in r.items() if k != "leaked_episode_ids"} for r in rows]
    with open(out / "overlap_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0]))
        w.writeheader()
        w.writerows(table_rows)
    (out / "overlap_table.md").write_text(_md_table(rows))
    print(f"[overlap] wrote {out / 'overlap_ids.json'}, overlap_table.{{csv,md}}")


if __name__ == "__main__":
    main()
