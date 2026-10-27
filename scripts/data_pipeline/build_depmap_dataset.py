"""Build the depmap_meta evidence-deliberation dataset (gene-essentiality domain, RQ4 generalization).

Target hypothesis: "gene G is a dependency (essential for fitness) in cell line C" (knockout reduces C's fitness).
Label: supported iff Chronos gene-effect <= -0.5 AND P(dependency) >= 0.5;
refuted iff Chronos > +0.25 (suppressor); else inconclusive.

Mirrors build_tahoe_meta_dataset.py design principles, reinterpreted for DepMap:
  P1  exclude common-essential (trivially supported) + non-expressed genes as targets
  P2  targets drawn from *selective* genes only (dependency in --selective-lo..--selective-hi of
      lines, default 2-50%) so the label depends on cell-line context; diversity across gene x lineage;
      50:50 supported:not; ~15% of negatives refuted (suppressors rare)
  P3  multi-modal relevance hierarchy, ~60:40 related:irrelevant, drawn top-first
  P4  sufficiency tiers (rich/moderate/poor) by count of conclusive related stubs
  P5  the target's own (G,C) CRISPR measurement is held out, never a stub

Splits follow the shared HuggingFace convention (see scripts/data_pipeline/README.md): a
stratified-iid train/val/test holdout written as one parquet file per split, with NO `split`
column (the file IS the split; loaded via `ibis_env_utils.load_dataset_split`). Gene/lineage
out-of-distribution axes are recorded as a descriptive `ood_group` column (`gene`/`lineage`/`iid`)
so analyses can break results down per axis — under iid splits the tag is a reporting grouping,
not a train-holdout.

Stubs are stored as structured dicts (modality, context_type, metadata, y) for the depmap_meta env to render.
Output: <out>/{train,val,test}.parquet + manifest.json in the unified schema.

Run via SLURM (it loads the full ~3 GB of DepMap/PRISM CSVs into memory; don't run on the
login node). The default 4000-episode build measured ~3 min wall and ~6.5 GB peak RSS, so
this is generous headroom:

  srun --partition=main --cpus-per-task=4 --mem=16G --time=00:30:00 \\
    .venv/bin/python scripts/data_pipeline/build_depmap_dataset.py \\
      --out-dir ibis_data/raw/depmap/episodes/v1
  # then: validate.py <out-dir>  and  publish.py --collection-dir <out-dir> --config depmap

The build is single-threaded (the work is CSV parsing, not compute), so extra CPUs don't
help — memory is the only real constraint. For a fast local check without the full source,
see the hermetic fixture build in scripts/data_pipeline/test_build_integration_depmap.py.
"""

import argparse
import json
import random
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DEPMAP_DIR = "ibis_data/raw/depmap/raw"
DEFAULT_PRISM_DIR = "ibis_data/raw/prism/raw"


def col_for(cols, sym):
    return next((c for c in cols if c.startswith(f"{sym} (")), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ibis_data/raw/depmap/episodes")
    ap.add_argument("--depmap-dir", default=DEFAULT_DEPMAP_DIR, help="dir with DepMap raw CSVs")
    ap.add_argument("--prism-dir", default=DEFAULT_PRISM_DIR, help="dir with PRISM raw CSVs")
    ap.add_argument("--n-episodes", type=int, default=4000)
    ap.add_argument("--pool-size", type=int, default=16)
    ap.add_argument("--relevant-fraction", type=float, default=0.6)
    ap.add_argument("--frac-supported", type=float, default=0.5)
    ap.add_argument("--frac-refuted-of-neg", type=float, default=0.15)
    # P4 sufficiency tiers: assigned by quota BEFORE pool construction (mirrors Tahoe's assign_tiers),
    # then build_pool fills supporting stubs up to the tier's quota and caps there.
    ap.add_argument("--frac-rich", type=float, default=1.0 / 3.0)
    ap.add_argument("--frac-poor", type=float, default=1.0 / 3.0)
    ap.add_argument("--rich-conclusive", type=int, default=4, help="# supporting stubs a 'rich' pool gets")
    ap.add_argument("--moderate-conclusive", type=int, default=1, help="# supporting stubs a 'moderate' pool gets")
    ap.add_argument("--holdout-lineages", type=int, default=3, help="# lineages tagged ood_group='lineage'")
    ap.add_argument("--holdout-gene-frac", type=float, default=0.12, help="fraction of genes tagged ood_group='gene'")
    ap.add_argument("--max-per-gene", type=int, default=12)
    ap.add_argument(
        "--selective-lo",
        type=float,
        default=0.02,
        help="P2: min fraction of measured lines a gene must be a dependency in to be a target",
    )
    ap.add_argument(
        "--selective-hi",
        type=float,
        default=0.50,
        help="P2: max such fraction (above this shades into trivially-supported common-essential)",
    )
    ap.add_argument("--test-size", type=int, default=800, help="# episodes held out as the test split")
    ap.add_argument("--val-size", type=int, default=256, help="# episodes held out as the val (monitoring) split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    D, P = Path(args.depmap_dir), Path(args.prism_dir)

    print("loading DepMap matrices ...")
    dep = pd.read_csv(D / "CRISPRGeneDependency.csv", index_col=0)
    eff = pd.read_csv(D / "CRISPRGeneEffect.csv", index_col=0)
    expr = pd.read_csv(D / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    cn = pd.read_csv(D / "OmicsCNGene.csv", index_col=0)
    hot = pd.read_csv(D / "OmicsSomaticMutationsMatrixHotspot.csv", index_col=0)
    dmg = pd.read_csv(D / "OmicsSomaticMutationsMatrixDamaging.csv", index_col=0)
    model = pd.read_csv(D / "Model.csv", index_col=0)
    gene_df = pd.read_csv(D / "Gene.csv").drop_duplicates("symbol").set_index("symbol")
    common_ess = set(g.split(" (")[0] for g in pd.read_csv(D / "AchillesCommonEssentialControls.csv")["Gene"])
    noness = [g.split(" (")[0] for g in pd.read_csv(D / "AchillesNonessentialControls.csv")["Gene"]]
    mlong = pd.read_csv(D / "OmicsSomaticMutations.csv", usecols=["ModelID", "HugoSymbol", "ProteinChange"])
    prot = {(r.ModelID, r.HugoSymbol): r.ProteinChange for r in mlong.itertuples(index=False)}  # O(1) lookup

    comp = pd.read_csv(P / "Repurposing_Public_24Q2_Extended_Primary_Compound_List.csv").dropna(
        subset=["repurposing_target"]
    )
    drug2cpd = {}
    for _, r in comp.iterrows():
        for g in str(r["repurposing_target"]).replace(";", ",").split(","):
            drug2cpd.setdefault(g.strip(), []).append((r["IDs"], r["Drug.Name"]))
    pm_path = P / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv"
    pm = pd.read_csv(pm_path, index_col=0)  # compounds x cells (ACH ids)
    portal = pd.read_csv(D / "PortalCompounds.csv")
    portal_by_name = {str(n).lower(): row for n, row in zip(portal["CompoundName"], portal.to_dict("records"))}

    lineage = model["OncotreeLineage"].to_dict()
    cellname = model["StrippedCellLineName"].fillna(pd.Series(model.index, index=model.index)).to_dict()
    sym2col = {c.split(" (")[0]: c for c in dep.columns}
    expr_cols = {c.split(" (")[0]: c for c in expr.columns}
    cn_cols = {c.split(" (")[0]: c for c in cn.columns}
    hot_cols = {c.split(" (")[0]: c for c in hot.columns}
    dmg_cols = {c.split(" (")[0]: c for c in dmg.columns}

    # standardized gene-effect matrix for fast co-essentiality (NaN -> column mean)
    E = eff.values.astype(np.float32)
    colmean = np.nanmean(E, axis=0)
    inds = np.where(np.isnan(E))
    E[inds] = np.take(colmean, inds[1])
    Z = (E - E.mean(0)) / (E.std(0) + 1e-6)
    gene_at = {c.split(" (")[0]: i for i, c in enumerate(eff.columns)}

    def partners(g, k=4):
        if g not in gene_at:
            return []
        cor = (Z[:, gene_at[g]] @ Z) / Z.shape[0]
        order = np.argsort(-cor)
        out = []
        for i in order:
            sym = eff.columns[i].split(" (")[0]
            if sym != g and sym not in common_ess:
                out.append(sym)
            if len(out) >= k:
                break
        return out

    @lru_cache(maxsize=None)
    def gene_meta(g):
        r = gene_df.loc[g] if g in gene_df.index else None
        return {
            "symbol": g,
            "name": (r["name"] if r is not None and pd.notna(r.get("name")) else None),
            "family": (r["gene_group"] if r is not None and pd.notna(r.get("gene_group")) else None),
            "ensembl_id": (
                r["ensembl_id"]
                if r is not None and "ensembl_id" in gene_df.columns and pd.notna(r.get("ensembl_id"))
                else None
            ),
        }

    @lru_cache(maxsize=None)
    def drivers_of(c, k=3):
        if c not in hot.index:
            return []
        row = hot.loc[c]
        return [col.split(" (")[0] for col in hot.columns if row[col] == 1][:k]

    @lru_cache(maxsize=None)
    def cell_meta(c):
        m = model.loc[c] if c in model.index else None
        return {
            "cell_name": cellname.get(c, c),
            "model_id": c,
            "lineage": lineage.get(c),
            "disease": (m.get("OncotreePrimaryDisease") if m is not None else None),
            "subtype": (m.get("OncotreeSubtype") if m is not None else None),
            "sex": (m.get("Sex") if m is not None else None),
            "primary_metastasis": (m.get("PrimaryOrMetastasis") if m is not None else None),
            "driver_mutations": drivers_of(c),
        }

    def crispr_y(g, c):
        gc = sym2col[g]
        return {"chronos": round(float(eff.at[c, gc]), 3), "p_dependency": round(float(dep.at[c, gc]), 3)}

    def label_of(g, c):
        y = crispr_y(g, c)
        if y["p_dependency"] >= 0.5 and y["chronos"] <= -0.5:
            return 1, None
        if y["chronos"] > 0.25:
            return 0, "refuted"
        return 0, "inconclusive"

    def expr_z(g, c):
        ec = expr_cols.get(g)
        if ec is None or c not in expr.index or pd.isna(expr.at[c, ec]):
            return None
        peers = expr.loc[[m for m in expr.index if lineage.get(m) == lineage.get(c)], ec].dropna()
        if len(peers) < 5:
            return None
        return round(float(expr.at[c, ec]), 2), round(float((expr.at[c, ec] - peers.mean()) / (peers.std() + 1e-9)), 2)

    def mol_stubs(g, c):
        out = []
        ez = expr_z(g, c)
        if ez:
            out.append(
                {
                    "modality": "expression",
                    "context_type": "molecular_expression",
                    "related": True,
                    "gene": g,
                    "gene_metadata": gene_meta(g),
                    "cell": cellname.get(c, c),
                    "cell_metadata": cell_meta(c),
                    "y": {"log2_tpm": ez[0], "lineage_z": ez[1]},
                }
            )
        cc = cn_cols.get(g)
        if cc and c in cn.index and pd.notna(cn.at[c, cc]):
            v = round(float(cn.at[c, cc]), 2)
            out.append(
                {
                    "modality": "copy_number",
                    "context_type": "molecular_cn",
                    "related": True,
                    "gene": g,
                    "gene_metadata": gene_meta(g),
                    "cell": cellname.get(c, c),
                    "cell_metadata": cell_meta(c),
                    "y": {"relative_cn": v, "call": ("amplified" if v > 1.5 else "deleted" if v < 0.5 else "neutral")},
                }
            )
        dc, hc = dmg_cols.get(g), hot_cols.get(g)
        is_d = bool(dc and c in dmg.index and dmg.at[c, dc] == 1)
        is_h = bool(hc and c in hot.index and hot.at[c, hc] == 1)
        pc = prot.get((c, g))
        out.append(
            {
                "modality": "mutation",
                "context_type": "molecular_mutation",
                "related": True,
                "gene": g,
                "gene_metadata": gene_meta(g),
                "cell": cellname.get(c, c),
                "cell_metadata": cell_meta(c),
                "y": {
                    "damaging": is_d,
                    "hotspot": is_h,
                    "protein_change": (pc if (is_d or is_h) else None),
                    "status": ("mutated" if (is_d or is_h) else "wild-type"),
                },
            }
        )
        return out

    def pharma_stub(g, c, via=None):
        if c not in pm.columns:
            return None
        cands = drug2cpd.get(g, [])
        hit = [(i, dn, float(pm.at[i, c])) for i, dn in cands if i in pm.index and pd.notna(pm.at[i, c])]
        if not hit:
            return None
        _, dn, v = min(hit, key=lambda x: x[2])
        prow = portal_by_name.get(str(dn).lower())
        dmeta = {
            "drug": dn,
            "targets": (prow.get("GeneSymbolOfTargets") if prow else g),
            "moa": (prow.get("TargetOrMechanism") if prow else None),
            "smiles": (prow.get("SMILES") if prow else None),
            "chembl_id": (prow.get("ChEMBLID") if prow else None),
        }
        return {
            "modality": "pharmacology",
            "context_type": "pharmacology",
            "related": True,
            "gene": g,
            "gene_metadata": gene_meta(g),
            "cell": cellname.get(c, c),
            "cell_metadata": cell_meta(c),
            "drug_metadata": dmeta,
            "via_pathway_gene": via,
            "y": {"viability_log2fc": round(v, 2), "sensitive": bool(v < -1)},
        }

    def crispr_stub(g, c, ctype, related, rho=None):
        s = {
            "modality": "crispr",
            "context_type": ctype,
            "related": related,
            "gene": g,
            "gene_metadata": gene_meta(g),
            "cell": cellname.get(c, c),
            "cell_metadata": cell_meta(c),
            "y": crispr_y(g, c),
        }
        if rho is not None:
            s["_coess_rho"] = round(rho, 2)  # hidden metadata for analysis; NOT rendered to the model
        return s

    def supporting(stub):
        """Evidence pointing TOWARD the gene being a dependency in this context (sufficiency, P4)."""
        y = stub["y"]
        if stub["modality"] == "crispr":
            return y["p_dependency"] >= 0.5
        if stub["modality"] == "expression":
            return y.get("lineage_z", 0) >= 1.5
        if stub["modality"] == "copy_number":
            return y["call"] == "amplified"
        if stub["modality"] == "mutation":
            return y["status"] == "mutated"
        if stub["modality"] == "pharmacology":
            return y["sensitive"]
        return False

    def build_pool(g, c, tier):
        # P4: the tier is a DESIGNED variable (assigned by quota in assign_tiers). We fill exactly
        # want_conc *supporting* relevant stubs (most-relevant modality first) and cap there, then top
        # up the relevant budget with NON-supporting relevant stubs — so a poor pool gets 0 supporting
        # stubs by construction and a rich pool gets rich_conclusive, mirroring Tahoe's build_pool.
        want_conc = {"rich": args.rich_conclusive, "moderate": args.moderate_conclusive, "poor": 0}[tier]
        gc = sym2col[g]
        lin = lineage.get(c)
        s = dep[gc].dropna().drop(index=c, errors="ignore")
        same = [m for m in s.index if lineage.get(m) == lin]
        other = [m for m in s.index if lineage.get(m) != lin]
        rng.shuffle(same)
        rng.shuffle(other)

        def sg_dep(m):  # would a same-gene stub in line m be SUPPORTING? (match supporting(): p_dep>=0.5)
            return float(dep.at[m, gc]) >= 0.5

        def sg(m):
            ct = "same_gene_same_lineage" if lineage.get(m) == lin else "same_gene_other_lineage"
            return crispr_stub(g, m, ct, True)

        same_pos, same_neg = [m for m in same if sg_dep(m)], [m for m in same if not sg_dep(m)]
        other_pos, other_neg = [m for m in other if sg_dep(m)], [m for m in other if not sg_dep(m)]
        # supporting same-gene candidates, most-relevant (same lineage) first; plenty for want_conc<=4
        sg_support = [sg(m) for m in same_pos[:8] + other_pos[:8]]
        sg_plain = [sg(m) for m in same_neg[:5] + other_neg[:3]]

        # other modalities: co-essential pathway, target-gene molecular state (expr/CN/mut), pharmacology
        pg = partners(g, 4)
        pathway = [
            crispr_stub(
                p, c, "coessential_pathway", True, rho=float((Z[:, gene_at[g]] @ Z[:, gene_at[p]]) / Z.shape[0])
            )
            for p in pg[:2]
            if p in sym2col and pd.notna(dep.at[c, sym2col[p]])
        ]
        molecular = mol_stubs(g, c)
        pharma = [x for x in [pharma_stub(g, c)] if x]
        for p in pg[:2]:
            pp = pharma_stub(p, c, via=g)
            if pp:
                pharma.append(pp)
                break
        other_mod = molecular + pharma + pathway
        mod_support = [x for x in other_mod if supporting(x)]
        mod_plain = [x for x in other_mod if not supporting(x)]

        n_rel = round(args.pool_size * args.relevant_fraction)
        # 1) exactly want_conc supporting relevant stubs (same-gene leads the relevance ladder), capped
        related = (sg_support + mod_support)[:want_conc]
        # 2) top up with NON-supporting relevant stubs, keeping same-gene first then multi-modal coverage
        for stub in sg_plain[:2] + mod_plain + sg_plain[2:]:
            if len(related) >= n_rel:
                break
            related.append(stub)
        controls = [
            crispr_stub(cg, c, "control", False)
            for cg in rng.sample(noness, min(12, len(noness)))
            if cg in sym2col and cg != g and pd.notna(dep.at[c, sym2col[cg]])
        ]
        pool = related + controls[: args.pool_size - len(related)]
        rng.shuffle(pool)
        return pool[: args.pool_size]

    # ---- target selection (balanced trichotomy + diversity + splits) ----
    print("selecting targets ...")
    dep_v, eff_v = dep.values, eff.values
    fin = ~(np.isnan(dep_v) | np.isnan(eff_v))
    gene_syms = np.array([c.split(" (")[0] for c in dep.columns])
    is_ce = np.array([g in common_ess for g in gene_syms])
    # P1/P2: restrict targets to *selective* dependency genes — those that are a dependency in
    # selective_lo..selective_hi of the lines they're measured in. Only for a selective gene does
    # the label genuinely depend on cell-line context (common-essential = trivially supported,
    # near-never-essential = trivially not), so this is what forces deliberation over recall.
    dep_call = fin & (dep_v >= 0.5) & (eff_v <= -0.5)
    dep_pos_line = fin & (dep_v >= 0.5)  # lines that yield a *supporting* same-gene stub (matches supporting())
    n_fin_per_gene = fin.sum(axis=0)
    frac_dep = np.divide(dep_call.sum(axis=0), n_fin_per_gene, out=np.zeros(len(gene_syms)), where=n_fin_per_gene > 0)
    selective = (frac_dep >= args.selective_lo) & (frac_dep <= args.selective_hi)
    keep_gene = (~is_ce) & selective
    sup_ceiling = int(np.minimum(args.max_per_gene, dep_call.sum(axis=0)[keep_gene]).sum())
    print(
        f"  selective target genes: {int(selective.sum())} (ex-CE: {int(keep_gene.sum())}) "
        f"of {len(gene_syms)}; supported-target ceiling ~= {sup_ceiling:,}"
    )
    sup_mask = dep_call & keep_gene[None, :]
    ref_mask = fin & (eff_v > 0.25) & keep_gene[None, :]
    inc_mask = fin & ~sup_mask & ~ref_mask & keep_gene[None, :]

    cells = np.array(dep.index)

    def pairs_from(mask, k, per_gene):
        # per_gene is shared across the supported/refuted/inconclusive calls so max_per_gene is a
        # GLOBAL cap (P2: no gene dominates the whole dataset, not just one class). A selective gene
        # legitimately recurs as both supported and inconclusive across different cell lines, but the
        # total per gene is still bounded by max_per_gene.
        ii, jj = np.where(mask)
        idx = nprng.permutation(len(ii))[: k * 8]  # oversample then cap per-gene
        out = []
        for t in idx:
            g, c = gene_syms[jj[t]], cells[ii[t]]
            if g not in sym2col or lineage.get(c) is None:
                continue
            if per_gene.get(g, 0) >= args.max_per_gene:
                continue
            per_gene[g] = per_gene.get(g, 0) + 1
            out.append((g, c))
            if len(out) >= k:
                break
        return out

    n = args.n_episodes
    n_sup = int(n * args.frac_supported)
    n_neg = n - n_sup
    n_ref = int(n_neg * args.frac_refuted_of_neg)
    n_inc = n_neg - n_ref
    per_gene = {}  # shared -> max_per_gene is a global cap across all three target classes
    targets = (
        pairs_from(sup_mask, n_sup, per_gene)
        + pairs_from(ref_mask, n_ref, per_gene)
        + pairs_from(inc_mask, n_inc, per_gene)
    )
    if len(targets) < n:
        print(
            f"  WARNING: produced {len(targets)} < requested {n} targets "
            f"(gene pool exhausted at max_per_gene={args.max_per_gene})"
        )
    rng.shuffle(targets)

    # ood_group tags (NOT the dataset split): mark a set of lineages and a fraction of genes so
    # analyses can break results down per generalization axis. The train/val/test split is iid.
    all_lineages = sorted({lineage.get(c) for _, c in targets if lineage.get(c)})
    holdout_lin = set(rng.sample(all_lineages, min(args.holdout_lineages, len(all_lineages))))
    all_genes = sorted({g for g, _ in targets})
    holdout_genes = set(rng.sample(all_genes, int(len(all_genes) * args.holdout_gene_frac)))

    # P4: assign sufficiency tiers BY QUOTA before building pools (mirrors Tahoe's assign_tiers).
    # A target's availability = # of other lines where the gene is itself a dependency (the dominant
    # source of supporting same-gene stubs). 'rich' is gated on availability >= rich_conclusive so the
    # pool can actually be filled to quota; poor/moderate are always achievable (they cap DOWN).
    def assign_tiers(targets):
        col_idx = {cc.split(" (")[0]: i for i, cc in enumerate(dep.columns)}
        cell_idx = {cell: i for i, cell in enumerate(cells)}
        ndep = dep_pos_line.sum(axis=0)  # supporting-eligible lines per gene (matches build_pool's sg_dep)
        avail = np.array([int(ndep[col_idx[g]]) - int(dep_pos_line[cell_idx[c], col_idx[g]]) for g, c in targets])
        nt = len(targets)
        n_rich, n_poor = round(args.frac_rich * nt), round(args.frac_poor * nt)
        order = np.argsort(-avail)
        tiers = ["moderate"] * nt
        r = 0
        for idx in order:
            if r >= n_rich:
                break
            if avail[idx] >= args.rich_conclusive:
                tiers[idx] = "rich"
                r += 1
        p = 0
        for idx in order[::-1]:
            if p >= n_poor:
                break
            if tiers[idx] == "moderate":
                tiers[idx] = "poor"
                p += 1
        print(
            f"  [tiers] assigned rich={tiers.count('rich')} moderate={tiers.count('moderate')} "
            f"poor={tiers.count('poor')} (rich gated by availability>={args.rich_conclusive})"
        )
        return tiers

    tiers = assign_tiers(targets)
    print(f"building {len(targets)} episodes ...")
    rows = []
    tier_mismatch = 0
    for (g, c), assigned_tier in zip(targets, tiers):
        label, ntype = label_of(g, c)
        pool = build_pool(g, c, assigned_tier)
        # store the tier the pool ACTUALLY realizes (should equal the assigned tier given the gating)
        nconc = sum(supporting(s) for s in pool if s["related"])
        tier = "rich" if nconc >= args.rich_conclusive else "moderate" if nconc >= args.moderate_conclusive else "poor"
        tier_mismatch += tier != assigned_tier
        lin = lineage.get(c)
        ood_group = "lineage" if lin in holdout_lin else "gene" if g in holdout_genes else "iid"
        rows.append(
            {
                "episode_id": str(uuid.uuid4())[:12],
                "target_gene": g,
                "target_cell": cellname.get(c, c),
                "target_cell_id": c,
                "target_direction": "essential",
                "label": int(label),
                "negative_type": ntype,
                "sufficiency_tier": tier,
                "lineage": lin,
                "ood_group": ood_group,
                "ground_truth": crispr_y(g, c),
                "target_gene_meta": gene_meta(g),
                "target_cell_meta": cell_meta(c),
                "context_pool": pool,
            }
        )

    if tier_mismatch:
        print(
            f"  [tiers] {tier_mismatch} pools realized a different tier than assigned "
            f"({tier_mismatch / len(rows):.1%}) — availability-limited; stored the realized tier"
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # stratified-iid splits: held-out test + small val for monitoring, rest train. The split is the
    # FILE (no `split` column) per the shared loader convention; ood_group stays as a metadata column.
    # Fail loudly *before* writing if the build came up short of test+val (target-sampling shortfall
    # or --n-episodes too small) — leaves at least one train episode.
    if len(df) <= args.test_size + args.val_size:
        raise ValueError(
            f"built {len(df)} episodes but --test-size ({args.test_size}) + --val-size ({args.val_size}) "
            f"= {args.test_size + args.val_size} need to fit with room for train; "
            "raise --n-episodes or lower --test-size/--val-size."
        )
    rest_df, test_df = train_test_split(df, test_size=args.test_size, stratify=df["label"], random_state=args.seed)
    train_df, val_df = train_test_split(
        rest_df, test_size=args.val_size, stratify=rest_df["label"], random_state=args.seed
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, split_df in splits.items():
        split_df.to_parquet(out / f"{split_name}.parquet", index=False, compression="zstd")

    manifest = {
        "version": "v1",
        "n_records": len(df),
        "seed": args.seed,
        "n_genes": int(df["target_gene"].nunique()),
        "n_cells": int(df["target_cell_id"].nunique()),
        "n_lineages": int(df["lineage"].nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
        "negative_type_counts": df["negative_type"].value_counts(dropna=False).to_dict(),
        "sufficiency_counts": df["sufficiency_tier"].value_counts().to_dict(),
        "ood_group_counts": df["ood_group"].value_counts().to_dict(),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "pool_size": args.pool_size,
        "mean_pool_len": float(df["context_pool"].map(len).mean()),
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ---- QC ----
    print(f"\nwrote train/val/test ({len(train_df)}/{len(val_df)}/{len(test_df)}) + manifest.json -> {out}")
    print(f"label: supported {100 * (df.label == 1).mean():.1f}% | not {(100 * (df.label == 0)).mean():.1f}%")
    print("negatives split: " + str(df[df.label == 0].negative_type.value_counts(normalize=True).round(3).to_dict()))
    print("sufficiency: " + str(df.sufficiency_tier.value_counts(normalize=True).round(3).to_dict()))
    print("ood_group: " + str(df.ood_group.value_counts().to_dict()))
    print(f"distinct target genes: {df.target_gene.nunique()} | lineages: {df.lineage.nunique()}")
    poolsz = df.context_pool.map(len)
    relfrac = df.context_pool.map(lambda p: np.mean([s["related"] for s in p]))
    print(
        f"pool size: mean {poolsz.mean():.1f} | related fraction: mean {relfrac.mean():.2f} (target {args.relevant_fraction})"
    )
    mods = pd.Series([s["modality"] for p in df.context_pool for s in p]).value_counts(normalize=True).round(3)
    print(f"stub modalities: {mods.to_dict()}")
    # leakage audit (P5): target (g, target_cell_id) must not appear as a stub
    leak = 0
    for r in rows:
        for s in r["context_pool"]:
            if (
                s["modality"] == "crispr"
                and s["gene"] == r["target_gene"]
                and s.get("cell_metadata", {}).get("model_id") == r["target_cell_id"]
            ):
                leak += 1
    print(f"P5 leakage (target row in own pool): {leak} (must be 0)")


if __name__ == "__main__":
    main()
