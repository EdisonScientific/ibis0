"""Generate the committed `depmap_mini` fixture for the hermetic build test.

Slices the real DepMap + PRISM raw CSVs down to a tiny, self-consistent subset (a handful
of cell lines concentrated in a few lineages, a few hundred genes spanning the label classes
+ control genes) so `build_depmap_dataset.py` runs end-to-end in seconds with no 3 GB source.
Run once against the real raw dirs; the output under `depmap_mini/` is committed.

    .venv/bin/python scripts/data_pipeline/fixtures/make_depmap_mini.py \
        --depmap-dir ibis_data/raw/depmap/raw \
        --prism-dir  ibis_data/raw/prism/raw
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CELLS = 40
N_LINEAGES = 4
N_SUPPORTED_GENES = 120
N_SUPPRESSOR_GENES = 40
N_OTHER_GENES = 60
N_CONTROL_GENES = 50  # nonessential controls (must be in the CRISPR matrix)
N_COMMON_ESS = 20


def _sym(col):
    return col.split(" (")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depmap-dir", default="ibis_data/raw/depmap/raw")
    ap.add_argument("--prism-dir", default="ibis_data/raw/prism/raw")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "depmap_mini"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    D, P = Path(args.depmap_dir), Path(args.prism_dir)
    out_d, out_p = Path(args.out_dir) / "depmap", Path(args.out_dir) / "prism"
    out_d.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)

    print("loading CRISPR matrices + model ...")
    dep = pd.read_csv(D / "CRISPRGeneDependency.csv", index_col=0)
    eff = pd.read_csv(D / "CRISPRGeneEffect.csv", index_col=0)
    model = pd.read_csv(D / "Model.csv", index_col=0)
    common_ess_all = [_sym(g) for g in pd.read_csv(D / "AchillesCommonEssentialControls.csv")["Gene"]]
    noness_all = [_sym(g) for g in pd.read_csv(D / "AchillesNonessentialControls.csv")["Gene"]]

    # --- choose cells: concentrate in a few lineages so expr_z has >=5 peers per lineage ---
    lineage = model["OncotreeLineage"].dropna()
    crispr_cells = [c for c in dep.index if c in eff.index and c in lineage.index]
    by_lin = pd.Series({c: lineage[c] for c in crispr_cells})
    top_lineages = by_lin.value_counts().head(N_LINEAGES).index.tolist()
    cells = []
    for lin in top_lineages:
        lin_cells = [c for c in crispr_cells if lineage[c] == lin]
        cells += lin_cells[: max(8, N_CELLS // N_LINEAGES)]
    cells = cells[:N_CELLS]
    print(f"  {len(cells)} cells across lineages {top_lineages}")

    # --- choose genes spanning the label classes (computed over the chosen cells) ---
    sym2col = {_sym(c): c for c in dep.columns}
    common_ess = set(common_ess_all)
    sub_dep = dep.loc[cells]
    sub_eff = eff.loc[cells]
    supported_genes, suppressor_genes = [], []
    for sym, col in sym2col.items():
        if sym in common_ess:
            continue
        d, e = sub_dep[col], sub_eff[col]
        if ((d >= 0.5) & (e <= -0.5)).any():
            supported_genes.append(sym)
        elif (e > 0.25).any():
            suppressor_genes.append(sym)
    rng.shuffle(supported_genes)
    rng.shuffle(suppressor_genes)

    other_pool = [s for s in sym2col if s not in common_ess]
    controls = [s for s in noness_all if s in sym2col]
    rng.shuffle(controls)
    common_in_matrix = [s for s in common_ess_all if s in sym2col]

    keep = set(supported_genes[:N_SUPPORTED_GENES])
    keep |= set(suppressor_genes[:N_SUPPRESSOR_GENES])
    keep |= set(rng.choice(other_pool, size=min(N_OTHER_GENES, len(other_pool)), replace=False))
    keep |= set(controls[:N_CONTROL_GENES])
    keep |= set(common_in_matrix[:N_COMMON_ESS])
    keep = {s for s in keep if s in sym2col}
    print(
        f"  {len(keep)} genes (supported>={min(N_SUPPORTED_GENES, len(supported_genes))}, "
        f"suppressor>={min(N_SUPPRESSOR_GENES, len(suppressor_genes))}, controls={len(controls[:N_CONTROL_GENES])})"
    )

    cells_set = set(cells)

    def _usecols(c):
        return c in ("", "Unnamed: 0") or _sym(c) in keep

    # --- matrices: dep/eff already loaded; expr/cn/hot/dmg re-read column-projected (memory-bounded) ---
    def write_matrix(df, name):
        cols = [c for c in df.columns if _sym(c) in keep]
        rows = [c for c in cells if c in df.index]
        df.loc[rows, cols].to_csv(out_d / name)

    write_matrix(dep, "CRISPRGeneDependency.csv")
    write_matrix(eff, "CRISPRGeneEffect.csv")
    for name in (
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "OmicsCNGene.csv",
        "OmicsSomaticMutationsMatrixHotspot.csv",
        "OmicsSomaticMutationsMatrixDamaging.csv",
    ):
        print(f"  projecting {name} ...")
        m = pd.read_csv(D / name, index_col=0, usecols=_usecols)
        rows = [c for c in cells if c in m.index]
        m.loc[rows].to_csv(out_d / name)

    # --- metadata + control lists, kept consistent with the chosen subset ---
    model.loc[[c for c in cells if c in model.index]].to_csv(out_d / "Model.csv")
    gene_df = pd.read_csv(D / "Gene.csv")
    gene_df[gene_df["symbol"].isin(keep)].to_csv(out_d / "Gene.csv", index=False)
    for name in ("AchillesCommonEssentialControls.csv", "AchillesNonessentialControls.csv"):
        full = pd.read_csv(D / name)
        full[full["Gene"].map(_sym).isin(keep)].to_csv(out_d / name, index=False)

    mlong = pd.read_csv(D / "OmicsSomaticMutations.csv", usecols=["ModelID", "HugoSymbol", "ProteinChange"])
    mlong[mlong["ModelID"].isin(cells_set) & mlong["HugoSymbol"].isin(keep)].to_csv(
        out_d / "OmicsSomaticMutations.csv", index=False
    )

    # --- PRISM: drugs targeting kept genes, matrix projected to chosen cells ---
    comp = pd.read_csv(P / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv")
    comp_keep = comp[
        comp["repurposing_target"]
        .fillna("")
        .apply(lambda t: any(g.strip() in keep for g in str(t).replace(";", ",").split(",")))
    ]
    comp_keep.to_csv(out_p / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv", index=False)
    pm = pd.read_csv(P / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv", index_col=0)
    pm_cells = [c for c in cells if c in pm.columns]
    pm_rows = [i for i in comp_keep["IDs"] if i in pm.index]
    pm.loc[pm_rows, pm_cells].to_csv(out_p / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv")
    drug_names = {str(n).lower() for n in comp_keep["Drug.Name"]}
    portal = pd.read_csv(D / "PortalCompounds.csv")
    portal[portal["CompoundName"].astype(str).str.lower().isin(drug_names)].to_csv(
        out_d / "PortalCompounds.csv", index=False
    )

    print(f"wrote fixture -> {args.out_dir}")
    print(f"  depmap: {sorted(p.name for p in out_d.glob('*.csv'))}")
    print(f"  prism:  {sorted(p.name for p in out_p.glob('*.csv'))}")


if __name__ == "__main__":
    main()
