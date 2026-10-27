"""Empirical predictive distribution over a candidate experiment's outcome, for BED.

The expected-information-gain policy needs, for a candidate experiment it has not yet run, a
distribution over the value that experiment would return. Rather than learn that distribution with a
(slow, and serving-unstable) generator model, we approximate it empirically: the plausible outcomes of
measuring gene $G$ are the real differential-expression values already observed for $G$ elsewhere in
the corpus. Sampling from this population is essentially free, so the Monte-Carlo sample count behind
each EIG estimate can be raised by 1-2 orders of magnitude, and no second model sits in the inner loop.

The held-out truth (the target's own (drug, cell) measurement of $G$) and the candidate's own
measurement are excluded so the predictive distribution never peeks at the answer.
"""

import collections

import pandas as pd
from ibis_env_utils import Split, load_dataset_split

# the outcome fields a candidate experiment "returns"; identity fields (gene, plate, ...) stay fixed.
VALUE_KEYS = ["log2FoldChange", "lfcSE", "padj", "baseMean", "n_cells_trt", "n_cells_ctrl"]


class ValuePopulation:
    """Per-gene population of real DE outcome-dicts, built once from the dataset parquet."""

    def __init__(self, data_path):
        # Built from train + val only: the BED experiment evaluates on the held-out `test`
        # split, so the empirical prior must never draw from it (no leakage into the policy).
        df = pd.concat(
            [load_dataset_split(data_path, s, columns=["context_pool"]) for s in (Split.TRAIN, Split.VAL)],
            ignore_index=True,
        )
        self.by_gene = collections.defaultdict(list)
        for cp in df["context_pool"]:
            if cp is None or isinstance(cp, float):
                continue
            for s in cp:
                y = s.get("y") or {}
                if s.get("kind") != "evidence" or y.get("log2FoldChange") is None:
                    continue
                gene = y.get("gene")
                if gene is None or not all(k in y for k in VALUE_KEYS):
                    continue
                drug = (s.get("drug_metadata") or {}).get("drug")
                cell = (s.get("cell_metadata") or {}).get("cell_name")
                self.by_gene[gene].append((drug, cell, {k: y[k] for k in VALUE_KEYS}))

    def sample(self, gene, n, rng, exclude=()):
        """`n` outcome-dicts drawn with replacement from gene `G`'s observed responses, dropping any
        whose (drug, cell) is in `exclude`. Empty list if the gene is unseen."""
        pool = [v for (d, c, v) in self.by_gene.get(gene, []) if (d, c) not in set(exclude)]
        return [rng.choice(pool) for _ in range(n)] if pool else []

    def stats(self):
        sizes = [len(v) for v in self.by_gene.values()]
        sizes.sort()
        med = sizes[len(sizes) // 2] if sizes else 0
        return {
            "genes": len(self.by_gene),
            "values": sum(sizes),
            "median_per_gene": med,
            "min_per_gene": (sizes[0] if sizes else 0),
        }
