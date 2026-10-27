"""Build the Tahoe Evidence Deliberation dataset from the HuggingFace source — one command.

Pipeline (single seeded entry point):

  download  : map the 1,026 cell-sharded DE files -> (cell, drug-set), pick a set of cells and a
              MATCHED set of drug-batches present across them, download just those shards (~matched
              drugs across many cells => dense same-drug-different-cell evidence).
  load/QC   : stream the shards; drop NaN-fit rows; compute is_significant + direction.
  filter    : drop artifact genes (mito/ribosomal/pseudogene) + baseMean<1 from TARGET candidates.
  sample    : stratified target sampling to hit, globally AND per cell line,
              50% supported / (1/3 refuted + 2/3 inconclusive) of the 50% not-supported.
              Significant events feed both supported (claim=observed dir) and refuted (claim=opposite).
  pools     : for each target assign a sufficiency tier (rich/moderate/poor, ~1/3 each, bounded by
              available conclusive evidence) and build a <=32-stub pool to that recipe, spanning the
              relevance ladder at ~60:40 relevant:irrelevant. Every stub annotated; no leakage.
  split     : standard stratified-iid train / val / test (random holdout over all episodes).
  write     : train.parquet + val.parquet + test.parquet + manifest.json (+ composition audit).

Splits are in-distribution: the random stratified holdout means train / val / test share the
same drug / gene / cell-line population. Out-of-distribution generalization (unseen cell
lineages, drug mechanisms, etc.) is evaluated with *separate* biological datasets (e.g. DepMap),
not by holding out a subset of combinations within this dataset.

Example (SLURM):
  srun --partition=main --cpus-per-task=16 --mem=128G --time=04:00:00 \\
    .venv/bin/python scripts/data_pipeline/build_tahoe_meta_dataset.py \\
      --download --n-cells 102 --n-drug-batches 2 --n-records 32000 \\
      --source-dir ibis_data/raw/tahoe-100m --output-dir ibis_data/raw/tahoe-meta-data/v2

Smoke (use already-downloaded shards, tiny N, no download; test/val must fit n-records):
  .venv/bin/python scripts/data_pipeline/build_tahoe_meta_dataset.py --n-records 400 \\
      --test-size 80 --val-size 40 --source-dir ibis_data/raw/tahoe-100m --output-dir /tmp/tahoe_smoke
"""

import argparse
import glob
import json
import re
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HF_REPO = "tahoebio/Tahoe-100M"
DE_PREFIX = "metadata/pseudobulk_differential_expression"
N_SHARDS = 1026
APPROX_BATCH_DRUGS = 64  # ~drugs per shard

# Genes whose differential expression is dominated by technical artifacts (P1).
ARTIFACT_RE = re.compile(
    r"^(MT-|MT[0-9]|MTRNR|MTATP|MTND|MTCO|MTCYB|RPL|RPS|RP[0-9]+-|MRPL|MRPS|"
    r"MALAT1$|NEAT1$|7SK$|XIST$|ENSG[0-9]+$)",
    re.IGNORECASE,
)

# relevance ladder (most -> least), for pool annotation
RELEVANCE_RUNGS = [
    "same_drug_diff_cell",
    "same_moa_evidence",
    "diff_drug_same_cell",
    "same_drug_cell_aggregate",
    "negative_control",
]


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source-dir",
        default="ibis_data/raw/tahoe-100m",
        help="Local dir with metadata parquets + de_shards/.../*.parquet",
    )
    p.add_argument("--output-dir", default="ibis_data/raw/tahoe-meta-data/v2")
    p.add_argument("--download", action="store_true", help="Map + download matched shards from HF")
    p.add_argument("--n-cells", type=int, default=102, help="Number of cell lines to include")
    p.add_argument("--n-drug-batches", type=int, default=2, help="Matched ~64-drug batches per cell")
    p.add_argument("--n-records", type=int, default=32000)
    p.add_argument("--pool-size", type=int, default=32)
    p.add_argument("--lfc-threshold", type=float, default=0.5)
    # class quotas (of all records)
    p.add_argument("--frac-supported", type=float, default=0.5)
    p.add_argument("--frac-refuted-of-neg", type=float, default=1.0 / 3.0)
    # sufficiency quotas (of all records)
    p.add_argument("--frac-rich", type=float, default=1.0 / 3.0)
    p.add_argument("--frac-poor", type=float, default=1.0 / 3.0)
    # pool recipe
    p.add_argument("--relevant-fraction", type=float, default=0.6)
    p.add_argument("--rich-conclusive", type=int, default=4, help="min conclusive relevant stubs for 'rich'")
    p.add_argument("--moderate-conclusive", type=int, default=1)
    p.add_argument("--test-size", type=int, default=2000, help="# episodes held out as the test split")
    p.add_argument("--val-size", type=int, default=256, help="# episodes held out as the val split")
    p.add_argument("--nonsig-per-cell", type=int, default=40000, help="non-sig rows kept per cell in pass 1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=16)
    return p.parse_args()


# ------------------------------------------------------------------- download + index
def _de_filename(idx):
    return f"train-{idx:05d}-of-{N_SHARDS:05d}.parquet"


def map_shard_remote(idx, fs):
    """Map a shard -> (cell, drug-set) from the parquet FOOTER only (~0.3s vs ~15s for a column read).

    Each shard is one cell line (constant column); drugs are sorted within the shard, so the union
    of per-row-group min/max statistics recovers the full ~64-drug set without reading any data.
    """
    import pyarrow.parquet as pq

    def s(v):
        return v.decode() if isinstance(v, bytes) else v

    path = f"datasets/{HF_REPO}/{DE_PREFIX}/{_de_filename(idx)}"
    with fs.open(path) as f:
        md = pq.read_metadata(f)
    names = list(md.schema.names)
    di, ci = names.index("drug"), names.index("Cell_Name_Vevo")
    drugs, cell = set(), None
    for rg in range(md.num_row_groups):
        dc = md.row_group(rg).column(di)
        if dc.is_stats_set and dc.statistics is not None:
            drugs.add(s(dc.statistics.min))
            drugs.add(s(dc.statistics.max))
        if cell is None:
            cc = md.row_group(rg).column(ci)
            if cc.is_stats_set and cc.statistics is not None:
                cell = s(cc.statistics.min)
    return idx, ([cell] if cell else []), drugs


def build_or_load_shard_index(out_dir, max_workers):
    """Map every shard -> (cell, drugs). Cached as shard_index.json (one-time, ~GB of column reads)."""
    cache = out_dir / "shard_index.json"
    if cache.exists():
        print(f"[index] using cached {cache}")
        raw = json.loads(cache.read_text())
        return {int(k): (v["cell"], set(v["drugs"])) for k, v in raw.items()}

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    print(f"[index] mapping {N_SHARDS} shards (cell+drug columns, {max_workers} workers)...")
    index = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(map_shard_remote, i, fs) for i in range(N_SHARDS)]
        for n, fut in enumerate(futs):
            idx, cells, drugs = fut.result()
            # a few boundary shards span 2 cells; assign the majority cell (first) for selection
            index[idx] = (cells[0], drugs)
            if (n + 1) % 100 == 0:
                print(f"[index]   {n + 1}/{N_SHARDS}  ({time.time() - t0:.0f}s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({str(k): {"cell": c, "drugs": sorted(d)} for k, (c, d) in index.items()}))
    print(f"[index] mapped + cached in {time.time() - t0:.0f}s")
    return index


def select_shards(index, n_cells, n_batches, seed):
    """Pick cells + a matched drug set present across them; return shard indices to fetch."""
    cells = sorted({c for c, _ in index.values()})
    # rank drugs by how many cells they appear in -> the most "matched" drugs
    drug_cells = defaultdict(set)
    for idx, (cell, drugs) in index.items():
        for d in drugs:
            drug_cells[d].add(cell)
    n_target_drugs = n_batches * APPROX_BATCH_DRUGS
    top_drugs = sorted(drug_cells, key=lambda d: -len(drug_cells[d]))[:n_target_drugs]
    top_drugs_set = set(top_drugs)
    chosen_cells = cells[:n_cells] if n_cells < len(cells) else cells

    # for each chosen cell, take the shards that cover the most of the matched drugs
    shards_to_get = set()
    for cell in chosen_cells:
        cell_shards = [(idx, drugs & top_drugs_set) for idx, (c, drugs) in index.items() if c == cell]
        cell_shards.sort(key=lambda x: -len(x[1]))
        covered, picked = set(), 0
        for idx, matched in cell_shards:
            if picked >= n_batches and len(covered) >= 0.8 * len(top_drugs_set):
                break
            if matched - covered:
                shards_to_get.add(idx)
                covered |= matched
                picked += 1
    print(f"[select] {len(chosen_cells)} cells, {len(top_drugs)} matched drugs, {len(shards_to_get)} shards")
    return sorted(shards_to_get), chosen_cells, top_drugs


def ensure_shards_local(shard_idxs, source_dir, max_workers=8):
    from huggingface_hub import hf_hub_download

    dest = Path(source_dir) / "de_shards"

    def fetch(idx):
        fname = _de_filename(idx)
        local = dest / DE_PREFIX / fname
        if not local.exists():
            hf_hub_download(HF_REPO, f"{DE_PREFIX}/{fname}", repo_type="dataset", local_dir=str(dest))
        return str(local)

    print(f"[download] fetching {len(shard_idxs)} shards ({max_workers} parallel)...", flush=True)
    t0 = time.time()
    got = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for n, p in enumerate(ex.map(fetch, shard_idxs)):
            got.append(p)
            if (n + 1) % 25 == 0:
                print(f"[download]   {n + 1}/{len(shard_idxs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[download] done in {time.time() - t0:.0f}s", flush=True)
    return got


# --------------------------------------------------------------------------- loading
def shard_paths(source_dir):
    d = Path(source_dir) / "de_shards" / DE_PREFIX
    if not d.exists():
        d = Path(source_dir) / "de_shards"
    return sorted(glob.glob(f"{d}/train-*.parquet"))


def is_artifact_gene(name):
    return bool(ARTIFACT_RE.match(str(name)))


def compute_verdict(lfc, padj, direction, thr):
    if lfc is None or padj is None or pd.isna(lfc) or pd.isna(padj):
        return "inconclusive"
    if padj >= 0.05 or abs(lfc) <= thr:
        return "inconclusive"
    obs = "upregulates" if lfc > 0 else "downregulates"
    return "supported" if obs == direction else "refuted"


def stream_pass1(paths, thr, nonsig_per_cell, seed):
    """One pass: collect significant rows, a per-cell non-sig subsample, and per-(drug,cell) aggregates."""
    rng = np.random.default_rng(seed)
    sig_parts, nonsig_parts = [], []
    nonsig_seen = Counter()
    agg = {}  # (drug,cell) -> dict
    cols = [
        "gene_name",
        "Cell_Name_Vevo",
        "drug",
        "concentration",
        "concentration_unit",
        "log2FoldChange",
        "padj",
        "baseMean",
        "n_cells_trt",
        "n_cells_ctrl",
    ]
    for p in paths:
        df = pd.read_parquet(p, columns=cols).rename(columns={"gene_name": "gene", "Cell_Name_Vevo": "cell"})
        df = df[df["padj"].notna() & df["log2FoldChange"].notna()]
        df["is_sig"] = (df["padj"] < 0.05) & (df["log2FoldChange"].abs() > thr)
        # aggregates per (drug, cell)
        for (drug, cell), g in df.groupby(["drug", "cell"], sort=False):
            a = agg.setdefault(
                (drug, cell), {"n_sig_genes_lfc05": 0, "n_sig_genes_lfc10": 0, "abs_lfc_sum": 0.0, "n": 0}
            )
            a["n_sig_genes_lfc05"] += int(g["is_sig"].sum())
            a["n_sig_genes_lfc10"] += int(((g["padj"] < 0.05) & (g["log2FoldChange"].abs() > 1.0)).sum())
            a["abs_lfc_sum"] += float(g["log2FoldChange"].abs().sum())
            a["n"] += len(g)
        sig_parts.append(df[df["is_sig"]])
        # subsample non-sig per cell
        ns = df[~df["is_sig"]]
        for cell, g in ns.groupby("cell", sort=False):
            room = nonsig_per_cell - nonsig_seen[cell]
            if room <= 0:
                continue
            take = g.sample(n=min(room, len(g)), random_state=int(rng.integers(2**31)))
            nonsig_parts.append(take)
            nonsig_seen[cell] += len(take)
    sig = pd.concat(sig_parts, ignore_index=True) if sig_parts else pd.DataFrame()
    nonsig = pd.concat(nonsig_parts, ignore_index=True) if nonsig_parts else pd.DataFrame()
    print(
        f"[pass1] significant rows: {len(sig):,} | non-sig sample: {len(nonsig):,} | (drug,cell) aggregates: {len(agg):,}"
    )
    return sig, nonsig, agg


def stream_pass2(paths, target_genes, thr):
    """Second pass: keep all rows whose gene is a target gene (stub neighbors are about the target gene)."""
    tg = set(target_genes)
    parts = []
    cols = [
        "gene_name",
        "Cell_Name_Vevo",
        "drug",
        "concentration",
        "concentration_unit",
        "log2FoldChange",
        "padj",
        "lfcSE",
        "stat",
        "pvalue",
        "baseMean",
        "n_cells_trt",
        "n_cells_ctrl",
        "plate",
    ]
    for p in paths:
        df = pd.read_parquet(p, columns=cols).rename(columns={"gene_name": "gene", "Cell_Name_Vevo": "cell"})
        df = df[df["gene"].isin(tg) & df["padj"].notna() & df["log2FoldChange"].notna()]
        for c in ("gene", "drug", "cell", "plate"):
            if c in df.columns:
                df[c] = df[c].astype("category")  # keep the big slice compact
        parts.append(df)
    slice_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    slice_df["is_sig"] = (slice_df["padj"] < 0.05) & (slice_df["log2FoldChange"].abs() > thr)
    print(f"[pass2] target-gene slice rows: {len(slice_df):,}")
    return slice_df


# ----------------------------------------------------------- target sampling (class x cell)
def classify_candidates(sig, nonsig, artifact_mask_col="_artifact"):
    """Return supported/refuted (from sig, via claimed direction) and inconclusive (from nonsig) candidate frames."""
    sig = sig[~sig["gene"].map(is_artifact_gene) & (sig["baseMean"] >= 1.0)].copy()
    nonsig = nonsig[~nonsig["gene"].map(is_artifact_gene) & (nonsig["baseMean"] >= 1.0)].copy()
    sig["obs_dir"] = np.where(sig["log2FoldChange"] > 0, "upregulates", "downregulates")
    return sig, nonsig


def per_cell_quotas(cells, n_records, frac_sup, frac_ref_of_neg):
    per = max(1, n_records // max(1, len(cells)))
    n_sup = round(per * frac_sup)
    n_neg = per - n_sup
    n_ref = round(n_neg * frac_ref_of_neg)
    n_inc = n_neg - n_ref
    return {"supported": n_sup, "refuted": n_ref, "inconclusive": n_inc}


def sample_targets(sig, nonsig, cells, quotas, thr, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for cell in cells:
        sc = sig[sig["cell"] == cell]
        nc = nonsig[nonsig["cell"] == cell]
        # supported: claim = observed direction
        sup = sc.sample(n=min(quotas["supported"], len(sc)), random_state=int(rng.integers(2**31))) if len(sc) else sc
        for _, r in sup.iterrows():
            rows.append((r, r["obs_dir"], "supported", 1, None))
        # refuted: a DIFFERENT significant row, claim = opposite of observed
        ref = sc.sample(n=min(quotas["refuted"], len(sc)), random_state=int(rng.integers(2**31))) if len(sc) else sc
        for _, r in ref.iterrows():
            opp = "downregulates" if r["obs_dir"] == "upregulates" else "upregulates"
            rows.append((r, opp, "refuted", 0, "refuted"))
        # inconclusive: non-sig, claim = random direction
        inc = (
            nc.sample(n=min(quotas["inconclusive"], len(nc)), random_state=int(rng.integers(2**31))) if len(nc) else nc
        )
        for _, r in inc.iterrows():
            d = rng.choice(["upregulates", "downregulates"])
            rows.append((r, d, "inconclusive", 0, "inconclusive"))
    rng.shuffle(rows)
    print(
        f"[sample] {len(rows):,} targets "
        f"(supported={sum(1 for x in rows if x[2] == 'supported')}, "
        f"refuted={sum(1 for x in rows if x[2] == 'refuted')}, "
        f"inconclusive={sum(1 for x in rows if x[2] == 'inconclusive')})"
    )
    return rows


# ----------------------------------------------------------- pool construction + tiers
def build_slice_index(slice_df):
    """gene -> DataFrame of (drug, cell, lfc, padj, is_sig, ...) for fast neighbor lookup."""
    return {g: gdf for g, gdf in slice_df.groupby("gene", sort=False)}


def dose_fields(row):
    # Keep dose metadata so structured baselines such as Rhaister can reconstruct treatment keys.
    dose = row.get("concentration")
    unit = row.get("concentration_unit")
    if pd.isna(dose) or pd.isna(unit):
        return {}
    return {"concentration": float(dose), "concentration_unit": str(unit)}


def stub_from_row(row, context_type, direction, thr, drug_meta, cell_meta, gene_meta):
    y = {
        "gene": row["gene"],
        "log2FoldChange": float(row["log2FoldChange"]),
        "padj": float(row["padj"]),
        "baseMean": float(row.get("baseMean", float("nan"))),
        **dose_fields(row),
    }
    for k in ("lfcSE", "stat", "pvalue", "n_cells_trt", "n_cells_ctrl", "plate"):
        if k in row and pd.notna(row[k]):
            y[k] = row[k].item() if hasattr(row[k], "item") else row[k]
    return {
        "kind": "evidence",
        "context_type": context_type,
        "drug_metadata": drug_meta,
        "cell_metadata": cell_meta,
        "gene_metadata": gene_meta,
        "y": y,
        "verdict": compute_verdict(row["log2FoldChange"], row["padj"], direction, thr),
    }


def assign_tiers(targets, gene_index, params):
    """Assign rich/moderate/poor. 'rich' needs >= rich_conclusive available conclusive relevant neighbors."""
    avail = []
    for i, (r, direction, klass, label, ntype) in enumerate(targets):
        gdf = gene_index.get(r["gene"])
        n_conc = 0
        if gdf is not None:
            # conclusive relevant neighbors = significant rows for this gene, other (drug,cell)
            nb = gdf[(gdf["is_sig"]) & ~((gdf["drug"] == r["drug"]) & (gdf["cell"] == r["cell"]))]
            n_conc = len(nb)
        avail.append(n_conc)
    avail = np.array(avail)
    n = len(targets)
    n_rich = round(params["frac_rich"] * n)
    n_poor = round(params["frac_poor"] * n)
    order = np.argsort(-avail)  # most conclusive-capable first
    tier = ["moderate"] * n
    # rich: top by availability that meet the threshold
    rich_assigned = 0
    for idx in order:
        if rich_assigned >= n_rich:
            break
        if avail[idx] >= params["rich_conclusive"]:
            tier[idx] = "rich"
            rich_assigned += 1
    # poor: lowest availability
    poor_assigned = 0
    for idx in order[::-1]:
        if poor_assigned >= n_poor:
            break
        if tier[idx] == "moderate":
            tier[idx] = "poor"
            poor_assigned += 1
    print(
        f"[tiers] rich={tier.count('rich')} moderate={tier.count('moderate')} poor={tier.count('poor')} "
        f"(rich requested {n_rich}; limited by availability)"
    )
    return tier


def build_pool(target_row, direction, tier, gene_index, agg, meta, params, rng):
    """Build a <=pool_size stub pool about the TARGET GENE, partitioned into a relevance ladder.

    Relevant = same-drug / same-MOA / same-cell neighbors of the target gene; irrelevant =
    unrelated drug x cell (negative controls). We fill ~relevant_fraction with relevant stubs
    (conclusive first, up to the tier's quota), one aggregate, then top up with negative controls.
    """
    drug, cell, gene = target_row["drug"], target_row["cell"], target_row["gene"]
    thr = params["lfc_threshold"]
    pool_size = params["pool_size"]
    n_relevant = int(round(params["relevant_fraction"] * pool_size))
    want_conc = {"rich": params["rich_conclusive"], "moderate": params["moderate_conclusive"], "poor": 0}[tier]
    target_moa = meta["moa"].get(drug)

    pool, used = [], {(drug, cell)}

    def add(row, ctype):
        key = (row["drug"], row["cell"])
        if key in used:
            return
        dm, cm, gm = meta["drug"].get(row["drug"], {}), meta["cell"].get(row["cell"], {}), meta["gene"].get(gene, {})
        pool.append(stub_from_row(row, ctype, direction, thr, dm, cm, gm))
        used.add(key)

    def n_conclusive():
        return sum(1 for s in pool if s["verdict"] in ("supported", "refuted"))

    gdf = gene_index.get(gene)
    cap = pool_size * 2  # plenty of candidates per rung; avoids iterating thousands of neighbors
    rungs = ("same_drug_diff_cell", "same_moa_evidence", "diff_drug_same_cell")
    conc_q = {rg: [] for rg in rungs}  # conclusive (significant) relevant neighbors, per rung
    inc_q = {rg: [] for rg in rungs}  # inconclusive relevant neighbors, per rung
    irr_rows = []  # unrelated drug x cell (negative controls)
    if gdf is not None and len(gdf):
        others = gdf[~((gdf["drug"] == drug) & (gdf["cell"] == cell))]
        moa = others["drug"].map(meta["moa"])
        rung_of = np.where(
            others["drug"].values == drug,
            "same_drug_diff_cell",
            np.where(
                (target_moa is not None) & (moa.values == target_moa),
                "same_moa_evidence",
                np.where(others["cell"].values == cell, "diff_drug_same_cell", "negative_control"),
            ),
        )
        others = others.assign(_rung=rung_of)

        def sample_rows(frame):
            if not len(frame):
                return []
            return frame.sample(n=min(cap, len(frame)), random_state=int(rng.integers(2**31))).to_dict("records")

        for rg in rungs:
            sub = others[others["_rung"] == rg]
            conc_q[rg] = sample_rows(sub[sub["is_sig"]])
            inc_q[rg] = sample_rows(sub[~sub["is_sig"]])
        irr_rows = sample_rows(others[others["_rung"] == "negative_control"])

    # 1) tier conclusive quota: most on-point rung first (rich gets ~want_conc; poor gets none)
    for rg in rungs:
        for row in conc_q[rg]:
            if n_conclusive() >= want_conc:
                break
            add(row, rg)
        if n_conclusive() >= want_conc:
            break
    # 2) round-robin INCONCLUSIVE relevant stubs to the relevant budget
    #    (keeps the ladder spectrum incl. same-drug WITHOUT flooding conclusive evidence)
    pos = {rg: 0 for rg in rungs}
    while len(pool) < n_relevant:
        progressed = False
        for rg in rungs:
            if len(pool) >= n_relevant:
                break
            while pos[rg] < len(inc_q[rg]):
                row = inc_q[rg][pos[rg]]
                pos[rg] += 1
                if (row["drug"], row["cell"]) not in used:
                    add(row, rg)
                    progressed = True
                    break
        if not progressed:
            break

    # 3) one aggregate for the target drug x cell, if available
    a = agg.get((drug, cell))
    if a and a["n"]:
        pool.append(
            {
                "kind": "aggregate",
                "context_type": "same_drug_cell_aggregate",
                "drug_metadata": meta["drug"].get(drug, {}),
                "cell_metadata": meta["cell"].get(cell, {}),
                "gene_metadata": {},
                "y": {
                    "n_sig_genes_lfc05": int(a["n_sig_genes_lfc05"]),
                    "n_sig_genes_lfc10": int(a["n_sig_genes_lfc10"]),
                    "mean_abs_lfc": float(a["abs_lfc_sum"] / max(a["n"], 1)),
                    "median_abs_lfc": 0.0,
                    "perturbation_strength_percentile": 0.0,
                },
                "verdict": None,
            }
        )

    # 4) irrelevant negative controls to fill the pool
    for row in irr_rows:
        if len(pool) >= pool_size:
            break
        add(row, "negative_control")
    rng.shuffle(pool)
    return pool[:pool_size]


# --------------------------------------------------------------------------------- write
def assemble_records(targets, tiers, gene_index, agg, meta, params, seed):
    rng = np.random.default_rng(seed + 1)
    records = []
    for (r, direction, klass, label, ntype), tier in zip(targets, tiers):
        pool = build_pool(r, direction, tier, gene_index, agg, meta, params, rng)
        cell_meta = meta["cell"].get(r["cell"], {})
        organ = cell_meta.get("Organ")
        records.append(
            {
                "episode_id": str(uuid.uuid4())[:12],
                "target_drug": r["drug"],
                "target_cell": r["cell"],
                "target_gene": r["gene"],
                "target_direction": direction,
                "label": int(label),
                "negative_type": ntype,
                "sufficiency_tier": tier,
                "lineage": organ,
                "ground_truth_de": {
                    "drug": r["drug"],
                    "cell_line": r["cell"],
                    "gene": r["gene"],
                    "log2FoldChange": float(r["log2FoldChange"]),
                    "padj": float(r["padj"]),
                    "baseMean": float(r.get("baseMean", float("nan"))),
                    "is_significant": bool(r.get("is_sig", False)),
                    "direction": r.get("obs_dir"),
                    **dose_fields(r),
                },
                "target_drug_meta": meta["drug"].get(r["drug"], {}),
                "target_cell_meta": cell_meta,
                "target_gene_meta": meta["gene"].get(r["gene"], {}),
                "context_pool": pool,
                "lfc_threshold": params["lfc_threshold"],
            }
        )
    return records


def load_metadata(source_dir):
    d = Path(source_dir)
    drug = pd.read_parquet(d / "drug_metadata.parquet")
    cell = pd.read_parquet(d / "cell_line_metadata.parquet").drop_duplicates("cell_name")
    gene = pd.read_parquet(d / "gene_metadata.parquet").drop_duplicates("gene_symbol")
    moa = dict(zip(drug["drug"], drug.get("moa-broad", pd.Series(dtype=str))))
    return {
        "drug": {
            r["drug"]: {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if pd.notna(v)}
            for _, r in drug.iterrows()
        },
        "cell": {
            r["cell_name"]: {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if pd.notna(v)}
            for _, r in cell.iterrows()
        },
        "gene": {
            r["gene_symbol"]: {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items() if pd.notna(v)}
            for _, r in gene.iterrows()
        },
        "moa": {k: v for k, v in moa.items() if pd.notna(v)},
    }


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.download:
        index = build_or_load_shard_index(out_dir, args.max_workers)
        shard_idxs, _, _ = select_shards(index, args.n_cells, args.n_drug_batches, args.seed)
        paths = ensure_shards_local(shard_idxs, args.source_dir)
    else:
        paths = shard_paths(args.source_dir)
    print(f"[load] {len(paths)} shards")

    meta = load_metadata(args.source_dir)
    sig, nonsig, agg = stream_pass1(paths, args.lfc_threshold, args.nonsig_per_cell, args.seed)
    sig, nonsig = classify_candidates(sig, nonsig)
    cells_all = sorted(set(sig["cell"]).union(nonsig["cell"]))
    q0 = per_cell_quotas(cells_all, args.n_records, args.frac_supported, args.frac_refuted_of_neg)
    sig_per_cell = sig["cell"].value_counts()
    # drop cells that can't supply their supported+refuted quota (keeps per-cell balance)
    cells = [c for c in cells_all if sig_per_cell.get(c, 0) >= q0["supported"] + q0["refuted"]][: args.n_cells]
    quotas = per_cell_quotas(cells, args.n_records, args.frac_supported, args.frac_refuted_of_neg)
    print(
        f"[quota] per-cell: {quotas}  over {len(cells)} cells (dropped {len(cells_all) - len(cells)} data-poor cells)",
        flush=True,
    )
    targets = sample_targets(sig, nonsig, cells, quotas, args.lfc_threshold, args.seed)

    target_genes = {t[0]["gene"] for t in targets}
    slice_df = stream_pass2(paths, target_genes, args.lfc_threshold)
    gene_index = build_slice_index(slice_df)

    params = vars(args)
    tiers = assign_tiers(targets, gene_index, params)

    records = assemble_records(targets, tiers, gene_index, agg, meta, params, args.seed)
    df = pd.DataFrame.from_records(records)

    # stratified-iid splits: held-out test + small val for monitoring, rest train
    rest_df, test_df = train_test_split(df, test_size=args.test_size, stratify=df["label"], random_state=args.seed)
    train_df, val_df = train_test_split(
        rest_df, test_size=args.val_size, stratify=rest_df["label"], random_state=args.seed
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, split_df in splits.items():
        split_df.to_parquet(out_dir / f"{split_name}.parquet", index=False, compression="zstd")

    manifest = {
        "version": "v2",
        "n_records": len(df),
        "seed": args.seed,
        "n_cells": int(df["target_cell"].nunique()),
        "n_drugs": int(df["target_drug"].nunique()),
        "n_genes": int(df["target_gene"].nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
        "negative_type_counts": df["negative_type"].value_counts(dropna=False).to_dict(),
        "sufficiency_counts": df["sufficiency_tier"].value_counts().to_dict(),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "pool_size": args.pool_size,
        "lfc_threshold": args.lfc_threshold,
        "mean_pool_len": float(df["context_pool"].map(len).mean()),
        "params": {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n[done] wrote train/val/test ({len(train_df)}/{len(val_df)}/{len(test_df)}) in {time.time() - t0:.0f}s")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
