import ast

import numpy as np
import pandas as pd

# Dose modes for the Rhaister treatment key.
#
# "require": a treatment is (drug, dose, unit), matching Rhaister's own
# definition, and a measurement without dose metadata is an error.
#
# "ignore" exists because the published episode datasets do not carry dose. `build_tahoe_meta_dataset.py`
# only emits `concentration`/`concentration_unit` when the raw Tahoe row has them (see `dose_fields`), and
# neither `EdisonScientific/EDB-Bio/tahoe` nor the test set built from it has those
# keys in `ground_truth_de` or in a stub's `y`. Recovering dose means matching every episode DE row back
# to the raw Tahoe shards (~16.5 GB), and it would change almost nothing:
#
#   - prompt-local: dose is never load-bearing. All stubs in an episode are about the target gene, and no
#     episode in the N=2000 test set repeats a (cell, drug) among its stubs (0/2000), so (cell, drug)
#     already identifies a Rhaister row uniquely. Dropping dose is exactly equivalent.
#   - pooled/all-context: 1.4% of fit rows share a (cell, drug) within a gene, and those get averaged
#     together instead of held apart as separate dose rows.
#
# The dose-free key is `[('drug',)]` rather than a (drug, 0.0, 'NA') triple so that it is visibly a
# different object in the predictions CSV: these numbers must not be mistaken for dose-ful ones.
DOSE_MODES = ("require", "ignore")
DOSE_FREE_UNIT = "unspecified"


def get_treatment_key(drug: str, measurement: dict, dose_mode: str = "require") -> str:
    """Format Tahoe drug+dose metadata into Rhaister's treatment key string."""
    if dose_mode not in DOSE_MODES:
        raise ValueError(f"dose_mode must be one of {DOSE_MODES}, got {dose_mode!r}")
    if dose_mode == "ignore":
        # Treatment collapses to the drug; every row for a drug gets a byte-identical key, which makes
        # `apply_max_predictable_dose` a no-op rather than a silent mis-selection.
        return str([(str(drug),)])
    missing = {"concentration", "concentration_unit"} - set(measurement)
    if missing:
        raise ValueError(
            f"Rhaister baselines require concentration metadata. Missing: {sorted(missing)}. "
            "The published episode datasets carry no dose; pass --dose-mode ignore to fit on "
            "(cell, drug) treatments instead."
        )
    dose = float(measurement["concentration"])
    unit = measurement["concentration_unit"]
    if pd.isna(dose) or pd.isna(unit):
        raise ValueError("Rhaister baselines require non-null concentration and concentration_unit.")
    return str([(str(drug), float(dose), str(unit))])


def parse_treatment(treatment: str) -> tuple[str, float, str]:
    """Extract (drug, dose, unit) from a Rhaister treatment key string.

    A dose-free key carries only the drug, and reports a NaN dose against `DOSE_FREE_UNIT` so that
    `apply_max_predictable_dose`'s `dose > best` comparison is always False and the single key for a drug
    stays selected.
    """
    parsed = ast.literal_eval(str(treatment))
    entry = parsed[0]
    if len(entry) == 1:
        return str(entry[0]), float("nan"), DOSE_FREE_UNIT
    drug, dose, unit = entry
    return str(drug), float(dose), str(unit)


def get_target_questions(episodes: pd.DataFrame, eval_split: str, dose_mode: str = "require") -> pd.DataFrame:
    """Build the per-episode question table (cell, drug, gene, direction, label) for a split."""
    rows = []
    for row in episodes[episodes["split"] == eval_split].to_dict("records"):
        rows.append(
            {
                "episode_id": row["episode_id"],
                "cell_line": str(row["target_cell_meta"].get("Cell_ID_Cellosaur") or row["target_cell"]),
                "target_cell": row["target_cell"],
                "drug": str(row["target_drug"]),
                "treatment": get_treatment_key(row["target_drug"], row["ground_truth_de"], dose_mode),
                "gene": str(row["target_gene"]),
                "target_direction": str(row["target_direction"]),
                "label": int(row["label"]),
            }
        )
    return pd.DataFrame(rows)


def make_measurement_row(cell_line: str, drug: str, gene: str, de: dict, dose_mode: str = "require") -> dict:
    """Convert one Tahoe DE result into the row schema used by Rhaister."""
    padj = float(de["padj"])
    pvalue = de.get("pvalue", padj)
    if pd.isna(pvalue):
        pvalue = padj
    reference = float(de["baseMean"])
    if pd.isna(reference):
        raise ValueError("Rhaister baselines require non-null baseMean/reference values.")
    return {
        "cell_line": str(cell_line),
        "treatment": get_treatment_key(drug, de, dose_mode),
        "gene": str(gene),
        "lfc": float(de["log2FoldChange"]),
        "pvalue": float(pvalue),
        "fdr": padj,
        "reference": reference,
    }


def get_target_measurements(episodes: pd.DataFrame, split: str, dose_mode: str = "require") -> pd.DataFrame:
    """Convert Tahoe-meta question rows into Rhaister DE measurements."""
    rows = []
    for row in episodes[episodes["split"] == split].to_dict("records"):
        cell_line = row["target_cell_meta"].get("Cell_ID_Cellosaur") or row["target_cell"]
        rows.append(
            make_measurement_row(cell_line, row["target_drug"], row["target_gene"], row["ground_truth_de"], dose_mode)
        )
    return pd.DataFrame(rows)


def get_context_measurements(episodes: pd.DataFrame, dose_mode: str = "require") -> pd.DataFrame:
    """Convert the evidence stubs shown in Tahoe-meta prompts into Rhaister rows."""
    rows = []
    for row in episodes.to_dict("records"):
        for stub in row["context_pool"]:
            if stub["kind"] != "evidence":
                continue
            y = stub["y"]
            cell_meta = stub["cell_metadata"]
            cell_line = cell_meta.get("Cell_ID_Cellosaur") or cell_meta["cell_name"]
            drug = stub["drug_metadata"]["drug"]
            gene = stub["gene_metadata"]["gene_symbol"]
            rows.append(make_measurement_row(cell_line, drug, gene, y, dose_mode))
    return pd.DataFrame(rows)


def to_rhaister_matrix(
    measurements: pd.DataFrame,
    row_index: pd.MultiIndex,
    genes: list[str],
    value: str,
) -> np.ndarray:
    """Pivot Tahoe-meta measurements into Rhaister's row-by-gene matrix format."""
    wide = measurements.pivot(index=["cell_line", "treatment"], columns="gene", values=value)
    wide = wide.reindex(index=row_index, columns=genes)
    if wide.isna().any().any():
        raise ValueError("Rhaister matrices cannot contain missing values.")
    return wide.to_numpy(dtype=np.float64)


def build_rhaister_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    genes: list[str] | None = None,
) -> dict:
    """Build Rhaister's explicit data dict from train and test measurements."""
    # Rhaister expects one value per (cell, treatment, gene), so average repeated observed rows before pivoting.
    value_cols = ["lfc", "pvalue", "fdr", "reference"]
    train = train.copy()
    train["pvalue"] = train["pvalue"].clip(1e-30, 1.0)
    train["fdr"] = train["fdr"].clip(1e-30, 1.0)
    train["reference"] = train["reference"].clip(lower=0.0)
    train = train.groupby(["cell_line", "treatment", "gene"], as_index=False)[value_cols].mean()

    if genes is None:
        genes = sorted(set(train["gene"].unique()))
    test = test[test["gene"].isin(genes)].copy()

    train_rows = pd.MultiIndex.from_frame(train[["cell_line", "treatment"]].drop_duplicates())
    test_rows = pd.MultiIndex.from_frame(test[["cell_line", "treatment"]].drop_duplicates())

    train_treatments = set(train_rows.get_level_values("treatment"))
    known_test = [treatment in train_treatments for treatment in test_rows.get_level_values("treatment")]
    if not all(known_test):
        test_rows = test_rows[np.asarray(known_test, dtype=bool)]

    train_cells = train_rows.get_level_values("cell_line").to_numpy(dtype=str)
    train_tr = train_rows.get_level_values("treatment").to_numpy(dtype=str)
    test_cells = test_rows.get_level_values("cell_line").to_numpy(dtype=str)
    test_tr = test_rows.get_level_values("treatment").to_numpy(dtype=str)

    data = {
        "train_cells": train_cells,
        "train_treatments": train_tr,
        "Y_train": to_rhaister_matrix(train, train_rows, genes, "lfc"),
        "P_train": to_rhaister_matrix(train, train_rows, genes, "pvalue"),
        "F_train": to_rhaister_matrix(train, train_rows, genes, "fdr"),
        "R_train": to_rhaister_matrix(train, train_rows, genes, "reference").astype(np.float32),
        "test_cells": test_cells,
        "test_treatments": test_tr,
        "gene_cols": genes,
        "n_test": len(test_rows),
    }
    # Rhaister needs test arrays for shape plumbing; Tahoe-meta targets are unlabeled prediction targets.
    data["Y_test"] = np.zeros((len(test_rows), len(genes)), dtype=np.float64)
    data["P_test"] = np.ones((len(test_rows), len(genes)), dtype=np.float64)
    data["F_test"] = np.ones((len(test_rows), len(genes)), dtype=np.float64)
    data["R_test"] = np.zeros((len(test_rows), len(genes)), dtype=np.float32)
    data["evaluate_test"] = lambda *args, **kwargs: {}
    return data


def get_fit_measurements(episodes: pd.DataFrame, args, test: pd.DataFrame) -> pd.DataFrame:
    """Build the fit measurements requested by the CLI flags."""
    dose_mode = getattr(args, "dose_mode", "require")
    train = get_target_measurements(episodes, args.train_split, dose_mode)
    if args.include_context_evidence:
        context = get_context_measurements(episodes, dose_mode)
        train = pd.concat([train, context], ignore_index=True)
    if args.exclude_eval_targets:
        eval_target_keys = set(
            zip(
                test["cell_line"].astype(str),
                test["treatment"].astype(str),
                test["gene"].astype(str),
                strict=True,
            )
        )
        keep = [
            (row.cell_line, row.treatment, row.gene) not in eval_target_keys for row in train.itertuples(index=False)
        ]
        train = train.loc[keep].copy()
    return train


def apply_max_predictable_dose(measurements: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Score dose-free Tahoe questions at the highest predictable same-unit dose."""
    if measurements.empty or train.empty:
        return measurements.copy()

    best_by_drug_unit: dict[tuple[str, str], tuple[float, str]] = {}
    for treatment in train["treatment"].dropna().astype(str).unique():
        drug, dose, unit = parse_treatment(treatment)
        key = (drug, unit)
        if key not in best_by_drug_unit or dose > best_by_drug_unit[key][0]:
            best_by_drug_unit[key] = (dose, treatment)

    out = measurements.copy()
    selected = []
    for treatment in out["treatment"].astype(str):
        drug, _dose, unit = parse_treatment(treatment)
        replacement = best_by_drug_unit.get((drug, unit))
        selected.append(treatment if replacement is None else replacement[1])

    out["treatment"] = selected
    return out
