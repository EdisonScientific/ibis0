"""Build the gwas_meta evidence-deliberation dataset (genetic-association domain, RQ4 generalization).

Target hypothesis: "SNP S (risk allele A) influences trait T in a stated direction of effect".
Label (spec-literal, applied to the held-out association):

    The hypothesis is SUPPORTED iff a held-out GWAS reports p < 5e-8 for this SNP-trait pair
    with a consistent direction of effect. Otherwise NOT SUPPORTED.

Concretely: each UNIQUE (SNP, trait) pair is one candidate hypothesis (the label's granularity;
~677K in the full catalog). The pair's own catalog association is the held-out ground truth and is
never a stub (P5, no leakage) — the evidence pool is built only from related-but-different
associations (tiers A-F below). When a pair has several study records, the most-significant one
(min p) is the representative held-out measurement and the hypothesis (allele, direction) is the
modal call across the pair's records. That held-out record makes the pair:
  supported                iff held-out p < 5e-8 AND direction consistent with the hypothesis
  not-supported / sub_threshold          iff held-out p >= 5e-8 (only suggestive; no genome-wide hit)
  not-supported / direction_inconsistent iff held-out p < 5e-8 but the strongest evidence flips direction
There are NO constructed negatives and no >=2-study requirement: positives are abundant, so the
balanced budget is limited by natural negatives — pairs whose best association is
only suggestive or direction-inconsistent. Balancing samples pairs from the three classes to hit
--frac-supported.

Mirrors build_depmap_dataset.py design principles, reinterpreted for the GWAS Catalog:
  P1  spec-literal label (p < 5e-8 + consistent direction); as targets we exclude MHC/HLA-region SNPs
      (artifact region), heavily-replicated 'textbook' pairs (studied in > --max-target-studies GWAS,
      i.e. recall not deliberation), and mixed-ancestry-ambiguous pairs --- all still admissible as evidence
  P2  diversity across gene x trait x EFO parent; per-gene AND per-trait caps; 50:50 supported:not
  P3  multi-tier relevance hierarchy (A-F), ~60:40 related:control, drawn most-relevant first
  P4  sufficiency tiers (rich/moderate/poor) ENGINEERED in ~equal thirds by capping how many informative
      (genome-wide-significant) related stubs each pool contains (rich >=4, moderate 1-3, poor 0)
  P5  the target's own (SNP, trait) replication record is held out, never a stub. LIMITATION: a same-gene
      or pathway stub on a different SNP in high LD with the target is effectively the same signal --- a
      residual leakage this check misses; filtering it needs an LD reference panel (deferred, documented)

Known characteristics (recorded in manifest.json):
  - related:control runs BELOW --relevant-fraction (real build ~0.32 vs 0.60 target). Unlike Tahoe
    (91% non-significant, so 'poor' pools are full of non-significant *related* stubs), the GWAS Catalog
    is a catalog of genome-wide-significant hits, so non-significant related evidence is scarce; the P4
    'poor' pools (~1/3 of episodes) therefore fill largely with unrelated controls. A future revision
    could build 'poor' from *indirect* related tiers (D/E) instead of controls.
  - ood_group can be trait-skewed (a few broad EFO parents dominate), so a handful of --holdout-traits
    parents may cover a large fraction of episodes. ood_group is a descriptive tag, not the split, so
    this does not affect the iid train/val/test partition; lower --holdout-traits for finer per-axis
    balance.

Relevance tiers (A-F -> context_type):
  A same_gene_same_trait      different SNP, same mapped gene, same trait
  B same_gene_related_trait   different SNP, same mapped gene, trait in the same EFO parent
  C same_snp_related_trait    same SNP, a different trait in the same EFO parent
  D same_snp_unrelated_trait  same SNP, a trait in a different EFO parent
  E pathway_gene_same_trait   SNP in a Reactome pathway-neighbor gene, associated with the target trait
  F control                   different SNP, different gene, unrelated trait

Splits follow the shared HuggingFace convention (see scripts/data_pipeline/README.md): a
stratified-iid train/val/test holdout written as one parquet file per split, with NO `split`
column (the file IS the split; loaded via `ibis_env_utils.load_dataset_split`). Gene / trait
generalization axes are preserved as a descriptive `ood_group` column (`gene`/`trait`/`iid`) so
analyses can break results down per axis, but the train/val/test split itself is iid.

Stubs are stored as structured dicts (context_type, metadata, y) for the gwas_meta env to render.
Output: <out>/{train,val,test}.parquet + manifest.json in the unified schema.

Inputs (raw dir laid out by scripts/data_pipeline/fetch_gwas_raw.py):
  <gwas-dir>/gwas_catalog_associations.tsv     NHGRI-EBI GWAS Catalog, associations w/ ontology annotations
  <gwas-dir>/gwas_trait_mappings.tsv           GWAS Catalog trait_mappings (EFO URI -> parent category); REQUIRED
  <gwas-dir>/Ensembl2Reactome.txt              Reactome gene->pathway (optional; enables tier E + enrichment)
  <gwas-dir>/gtex_gene_median_tpm.gct.gz       GTEx v8 median TPM per gene (optional; enrichment only)

EFO parent categories come from the Catalog's own published trait_mappings file (its
'Parent term' column), a required, deterministic build input: the build fails loudly if it is
missing rather than silently changing behavior, so every run is reproducible. Reactome / GTEx remain
optional enrichments (absent, that enrichment is simply skipped, mirroring depmap). Run via SLURM
(the full catalog is ~1M rows); for a fast local check without the real source see the hermetic
fixture build in scripts/data_pipeline/test_build_integration_gwas.py.

  srun --partition=main --cpus-per-task=4 --mem=32G --time=00:30:00 \\
    .venv/bin/python scripts/data_pipeline/build_gwas_dataset.py \\
      --out-dir ibis_data/raw/gwas/episodes/v1
  # then: validate.py <out-dir>  and  publish.py --collection-dir <out-dir> --config gwas
"""

import argparse
import gzip
import json
import random
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_GWAS_DIR = "ibis_data/raw/gwas/raw"
GW_SIG = 5e-8  # genome-wide significance threshold (the label bar)
# Extended MHC/HLA region on chr6 (GRCh38): its extreme, long-range LD makes single-variant labels
# unreliable, so it is excluded from the TARGET set (P1, artifact-region analog). MHC variants may
# still appear as evidence stubs, exactly as Tahoe/DepMap keep artifact genes admissible as evidence.
MHC_CHR, MHC_START, MHC_END = "6", 25_000_000, 34_000_000
_BASES = {"A", "C", "G", "T"}
_ANCESTRIES = [
    "European",
    "East Asian",
    "South Asian",
    "African American",
    "African",
    "Hispanic",
    "Latin American",
    "Native American",
    "Sub-Saharan African",
    "Greater Middle Eastern",
    "Oceanian",
    "Asian",
]


def _s(x):
    """Coerce a cell to a stripped string, mapping NaN / blank to None (dtype=str leaves gaps as NaN)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    s = str(x).strip()
    return s or None


def _f(x):
    """Parse a float, tolerating blanks / 'NR' / stray text; None on failure."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def parse_allele(strongest):
    """Risk allele from 'STRONGEST SNP-RISK ALLELE' (e.g. 'rs7903146-T' -> 'T'). None if unknown ('?')."""
    if not isinstance(strongest, str) or "-" not in strongest:
        return None
    a = strongest.rsplit("-", 1)[1].strip().upper()
    return a if a in _BASES else None


def in_mhc(chrom, pos):
    """Is (chrom, pos) inside the extended MHC/HLA region on chr6 (GRCh38)? Tolerates non-numeric pos."""
    try:
        return chrom == MHC_CHR and MHC_START <= int(pos) <= MHC_END
    except (TypeError, ValueError):
        return False


def effect_direction(or_beta, ci_text):
    """Direction the RISK ALLELE moves the trait: +1 increase, -1 decrease, None if unknown.

    Prefers the '95% CI (TEXT)' wording ('unit increase' / 'unit decrease') which carries the beta
    sign; falls back to the odds ratio (OR > 1 -> increase, OR < 1 -> decrease for the risk allele).
    """
    t = (ci_text or "").lower()
    if "increase" in t:
        return 1
    if "decrease" in t:
        return -1
    if or_beta is not None and or_beta > 0:
        if or_beta > 1.0:
            return 1
        if or_beta < 1.0:
            return -1
    return None


def direction_phrase(allele, direction, trait):
    """Plain-language direction sentence for a stub."""
    if allele is None or direction is None:
        return None
    verb = "increases" if direction > 0 else "decreases"
    return f"{allele} allele {verb} {trait}"


def consistent(allele, direction, ref_allele, ref_direction):
    """Is a record's (allele, direction) consistent with the pooled hypothesis (ref_allele, ref_direction)?

    Same allele -> same direction; the opposite reported allele -> flipped direction (allele-flip). If
    direction is unknown for either side, fall back to allele agreement (conservative).
    """
    if direction is None or ref_direction is None:
        return allele == ref_allele
    if allele == ref_allele:
        return direction == ref_direction
    return direction == -ref_direction


def parse_ancestry(sample_text):
    """Coarse ancestry from the free-text 'INITIAL SAMPLE SIZE' field (first ancestry keyword hit)."""
    if not isinstance(sample_text, str):
        return None
    for a in _ANCESTRIES:
        if a.lower() in sample_text.lower():
            return a
    return None


def parse_sample_size(sample_text):
    """Sum the comma-formatted integers in the 'INITIAL SAMPLE SIZE' free text; None if none found."""
    if not isinstance(sample_text, str):
        return None
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]{2,}", sample_text) if n.replace(",", "").isdigit()]
    return sum(nums) if nums else None


def load_reactome(path):
    """Ensembl2Reactome.txt -> (gene_ensembl -> set(pathway_id), pathway_id -> pathway_name).

    Human rows only. Returns ({}, {}) if the file is absent.
    """
    gene_pathways, pathway_name = defaultdict(set), {}
    if not path.exists():
        return gene_pathways, pathway_name
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["ens_id", "pathway_id", "url", "pathway_name", "evidence", "species"],
        dtype=str,
    )
    df = df[df["species"] == "Homo sapiens"]
    for r in df.itertuples(index=False):
        gene_pathways[r.ens_id].add(r.pathway_id)
        pathway_name[r.pathway_id] = r.pathway_name
    return gene_pathways, pathway_name


def load_gtex(path):
    """GTEx median-TPM .gct(.gz) -> ensembl_id (versionless) -> "Tissue TPM, Tissue TPM, ..." top-3 string."""
    top = {}
    if not path.exists():
        return top
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        fh.readline()  # '#1.2'
        fh.readline()  # dims
        header = fh.readline().rstrip("\n").split("\t")
        tissues = header[2:]
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ens = parts[0].split(".")[0]
            vals = np.array([_f(x) or 0.0 for x in parts[2:]], dtype=float)
            order = np.argsort(-vals)[:3]
            top[ens] = ", ".join(f"{tissues[i]} {vals[i]:.1f} TPM" for i in order if vals[i] > 0)
    return top


def load_trait_parents(path):
    """GWAS Catalog trait_mappings TSV -> {EFO URI: parent-category term}.

    Reuse the Catalog's own published trait categories rather than re-deriving them from
    the raw EFO ontology. The trait_mappings download has columns 'Disease trait', 'EFO term',
    'EFO URI', 'Parent term', 'Parent URI'; we key the parent category by EFO URI so it joins
    directly onto each association's MAPPED_TRAIT_URI. When a URI maps to more than one parent the
    alphabetically-first is chosen so the result is independent of row order (reproducibility).

    This file is a REQUIRED build input: EFO-parent grouping drives the relevance tiers (B/C/D), so
    the build fails loudly if it is missing rather than silently degrading to a different behavior.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"required trait-parent map not found: {path}\n"
            "Fetch it with scripts/data_pipeline/fetch_gwas_raw.py (GWAS Catalog trait_mappings)."
        )
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower().strip(): c for c in df.columns}  # tolerate minor header/casing variation
    uri_col = cols.get("efo uri") or cols.get("efo_uri")
    par_col = cols.get("parent term") or cols.get("parent")
    if not uri_col or not par_col:
        raise ValueError(f"{path}: expected 'EFO URI' and 'Parent term' columns, got {list(df.columns)}")
    multi = defaultdict(set)
    for _, r in df.iterrows():
        u, p = _s(r[uri_col]), _s(r[par_col])
        if u and p:
            multi[u].add(p)
    return {u: sorted(ps)[0] for u, ps in multi.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ibis_data/raw/gwas/episodes")
    ap.add_argument("--gwas-dir", default=DEFAULT_GWAS_DIR, help="dir with the GWAS Catalog + enrichment files")
    ap.add_argument("--assoc-file", default="gwas_catalog_associations.tsv", help="associations TSV filename")
    ap.add_argument(
        "--trait-mappings",
        default="gwas_trait_mappings.tsv",
        help="GWAS Catalog trait_mappings TSV (EFO URI -> parent category); REQUIRED",
    )
    ap.add_argument(
        "--n-episodes",
        type=int,
        default=32000,
        help="episode budget. Default 32000 matches the Tahoe build (--n-records 32000); bounded above by "
        "the ~231K natural-negative ceiling (one candidate per unique SNP-trait)",
    )
    ap.add_argument("--pool-size", type=int, default=16)
    ap.add_argument("--relevant-fraction", type=float, default=0.6)
    ap.add_argument("--frac-supported", type=float, default=0.5)
    ap.add_argument(
        "--frac-direction-inconsistent-of-neg",
        type=float,
        default=0.15,
        help="fraction of negatives that are direction-inconsistent (vs sub-threshold); direction flips are rarer",
    )
    ap.add_argument("--holdout-traits", type=int, default=8, help="# EFO parents tagged ood_group='trait'")
    ap.add_argument("--holdout-gene-frac", type=float, default=0.12, help="fraction of genes tagged ood_group='gene'")
    ap.add_argument("--max-per-gene", type=int, default=12, help="P2: cap episodes per target gene")
    ap.add_argument(
        "--max-per-trait",
        type=int,
        default=40,
        help="P2: cap episodes per target trait so mega-traits (height, BMI, ...) don't dominate",
    )
    ap.add_argument(
        "--max-target-studies",
        type=int,
        default=25,
        help="P1: exclude heavily-replicated 'textbook' SNP-trait pairs (studied in > this many GWAS) "
        "as targets, so the answer needs deliberation over pooled evidence rather than recall; they "
        "may still appear as evidence",
    )
    ap.add_argument("--test-size", type=int, default=2000, help="# episodes held out as the test split (Tahoe: 2000)")
    ap.add_argument("--val-size", type=int, default=256, help="# episodes held out as the val (monitoring) split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    G = Path(args.gwas_dir)

    print("loading GWAS Catalog associations ...")
    assoc = pd.read_csv(G / args.assoc_file, sep="\t", dtype=str, low_memory=False)
    gene_pathways, pathway_name = load_reactome(G / "Ensembl2Reactome.txt")
    gtex_top = load_gtex(G / "gtex_gene_median_tpm.gct.gz")
    uri_parent = load_trait_parents(G / args.trait_mappings)  # required; raises if absent
    print(
        f"  reactome genes: {len(gene_pathways)} | gtex genes: {len(gtex_top)} | trait-parent URIs: {len(uri_parent)}"
    )

    def efo_parent_of(uri_field):
        """Parent category for an association's MAPPED_TRAIT_URI (first mapped EFO URI wins)."""
        if not uri_field:
            return "Uncategorized"
        for u in re.split(r"[;,]", uri_field):
            p = uri_parent.get(u.strip())
            if p:
                return p
        return "Uncategorized"

    # ---- parse + filter to valid, single-gene records ----
    def is_single_gene(gene_ids, mapped_gene):
        return (
            gene_ids is not None
            and "," not in gene_ids
            and mapped_gene is not None
            and "," not in mapped_gene
            and " x " not in mapped_gene  # interaction annotations
        )

    records = []
    for d in assoc.to_dict(orient="records"):  # preserves exact column names (spaces/hyphens/%)
        snp = _s(d.get("SNPS"))
        trait = _s(d.get("MAPPED_TRAIT"))
        if not snp or not snp.startswith("rs") or ";" in snp or "," in snp or " x " in snp:
            continue  # single rsID only
        if not trait or not _s(d.get("MAPPED_TRAIT_URI")):
            continue
        gene_ids, mapped_gene = _s(d.get("SNP_GENE_IDS")), _s(d.get("MAPPED_GENE"))
        if not is_single_gene(gene_ids, mapped_gene):
            continue
        p = _f(d.get("P-VALUE"))
        if p is None or not (0 < p < 1):
            continue
        allele = parse_allele(_s(d.get("STRONGEST SNP-RISK ALLELE")))
        or_beta = _f(d.get("OR or BETA"))
        ci = _s(d.get("95% CI (TEXT)"))
        records.append(
            {
                "snp": snp,
                "trait": trait,
                "trait_uri": _s(d.get("MAPPED_TRAIT_URI")),
                "efo_parent": efo_parent_of(_s(d.get("MAPPED_TRAIT_URI"))),
                "gene": mapped_gene,
                "gene_ensembl": gene_ids,
                "chr": _s(d.get("CHR_ID")),
                "pos": _s(d.get("CHR_POS")),
                "allele": allele,
                "raf": _f(d.get("RISK ALLELE FREQUENCY")),
                "functional_class": _s(d.get("CONTEXT")),
                "intergenic": _s(d.get("INTERGENIC")) == "1",
                "study": _s(d.get("STUDY ACCESSION")),
                "ancestry": parse_ancestry(_s(d.get("INITIAL SAMPLE SIZE"))),
                "sample_size": parse_sample_size(_s(d.get("INITIAL SAMPLE SIZE"))),
                "platform": _s(d.get("PLATFORM [SNPS PASSING QC]")) or _s(d.get("GENOTYPING TECHNOLOGY")),
                "p_value": p,
                "or_beta": or_beta,
                "ci_text": ci,
                "direction": effect_direction(or_beta, ci),
            }
        )
    print(f"  valid single-gene records: {len(records)} (of {len(assoc)})")
    if not records:
        raise ValueError("no valid single-gene records — check the associations file / column names")

    # ---- indexes ----
    by_snp_trait = defaultdict(list)  # (snp, trait) -> records  (the hypothesis granularity)
    by_gene_trait = defaultdict(list)  # (gene, trait) -> records
    by_gene = defaultdict(list)  # gene -> records
    by_snp = defaultdict(list)  # snp -> records
    by_trait = defaultdict(list)  # trait -> records
    by_parent_traits = defaultdict(set)  # efo_parent -> set(trait)
    gene_of = {}  # snp -> gene (single-gene, so stable)
    for rec in records:
        by_snp_trait[(rec["snp"], rec["trait"])].append(rec)
        by_gene_trait[(rec["gene"], rec["trait"])].append(rec)
        by_gene[rec["gene"]].append(rec)
        by_snp[rec["snp"]].append(rec)
        by_trait[rec["trait"]].append(rec)
        by_parent_traits[rec["efo_parent"]].add(rec["trait"])
        gene_of[rec["snp"]] = rec["gene"]
    all_genes = list(by_gene)

    # gene -> pathway-neighbor genes (share >=1 Reactome pathway), for tier E
    ens_of_gene = {rec["gene"]: rec["gene_ensembl"] for rec in records}
    pathway_members = defaultdict(set)  # pathway_id -> set(gene symbol)
    for g, ens in ens_of_gene.items():
        for pid in gene_pathways.get(ens, ()):
            pathway_members[pid].add(g)

    def pathway_neighbors(gene, k=8):
        ens = ens_of_gene.get(gene)
        out = set()
        for pid in gene_pathways.get(ens, ()):
            out |= pathway_members[pid]
        out.discard(gene)
        return list(out)[:k]

    def gene_meta(gene, ens=None):
        ens = ens or ens_of_gene.get(gene)
        pids = sorted(gene_pathways.get(ens, ()))
        return {
            "symbol": gene,
            "ensembl_id": ens,
            "reactome_pathways": [pathway_name.get(p, p) for p in pids[:6]],
            "gtex_top_tissues": gtex_top.get((ens or "").split(".")[0]),
        }

    def stub(rec, context_type, related, target_pos=None):
        """Structured evidence stub (rendered by the gwas_meta env)."""
        s = {
            "modality": "association",
            "context_type": context_type,
            "related": related,
            "snp": rec["snp"],
            "gene": rec["gene"],
            "trait": rec["trait"],
            "snp_metadata": {
                "rsid": rec["snp"],
                "chr": rec["chr"],
                "pos": rec["pos"],
                "risk_allele": rec["allele"],
                "raf": rec["raf"],
                "functional_class": rec["functional_class"],
                "intergenic": rec["intergenic"],
                "distance_to_target": (
                    abs(int(rec["pos"]) - int(target_pos))
                    if target_pos and rec["pos"] and str(rec["pos"]).isdigit() and str(target_pos).isdigit()
                    else None
                ),
            },
            "gene_metadata": gene_meta(rec["gene"], rec["gene_ensembl"]),
            "trait_metadata": {"trait": rec["trait"], "efo_parent": rec["efo_parent"]},
            "study_metadata": {
                "accession": rec["study"],
                "ancestry": rec["ancestry"],
                "sample_size": rec["sample_size"],
                "platform": rec["platform"],
            },
            "y": {
                "p_value": rec["p_value"],
                "or_beta": rec["or_beta"],
                "ci_text": rec["ci_text"],
                "direction": rec["direction"],
                "direction_phrase": direction_phrase(rec["allele"], rec["direction"], rec["trait"]),
            },
        }
        return s

    def informative(s):
        """Evidence pointing TOWARD the hypothesis (genome-wide-significant), for sufficiency (P4)."""
        return s["y"]["p_value"] is not None and s["y"]["p_value"] < GW_SIG

    # ---- candidate (SNP, trait) hypotheses ----
    # One candidate per UNIQUE (SNP, trait) hypothesis (~677K in the full catalog).
    # The pair's own catalog association IS the held-out ground truth (never a stub — P5); the label is
    # spec-literal on it: SUPPORTED iff a held-out GWAS reports p < 5e-8 with a consistent direction of
    # effect, otherwise NOT SUPPORTED. When a pair has several study records, the most-significant one
    # (min p) is the representative held-out measurement, and the hypothesis (allele, direction) is the
    # modal call across the pair's records. No >=2-study requirement: positives are abundant, so the
    # balanced budget is bounded by the NATURAL negatives (pairs whose best association is only
    # suggestive, i.e. p >= 5e-8, or whose strongest evidence flips direction).
    print("selecting candidate SNP-trait hypotheses ...")
    candidates = []
    n_drop_mhc = n_drop_repl = 0
    for (snp, trait), recs in by_snp_trait.items():
        gene = gene_of.get(snp, recs[0]["gene"])
        held = min(recs, key=lambda r: r["p_value"])  # representative held-out replication (most significant)

        # P1 (artifact region): exclude MHC/HLA-region SNPs as targets (kept as evidence elsewhere).
        if in_mhc(held["chr"], held["pos"]):
            n_drop_mhc += 1
            continue
        # P1 (deliberation over recall): exclude heavily-replicated 'textbook' pairs as targets.
        if len({r["study"] for r in recs if r["study"]}) > args.max_target_studies:
            n_drop_repl += 1
            continue

        # mixed-ancestry ambiguity (P1): significant in some tested ancestries, sub-threshold in others.
        anc_sig = defaultdict(list)
        for rec in recs:
            if rec["ancestry"]:
                anc_sig[rec["ancestry"]].append(rec["p_value"] < GW_SIG)
        if len(anc_sig) >= 2:
            per_anc = {a: any(v) for a, v in anc_sig.items()}
            if any(per_anc.values()) and not all(per_anc.values()):
                continue  # mixed evidence across ancestries -> drop

        # hypothesis (allele, direction) = modal call across the pair's records
        alleles = [rec["allele"] for rec in recs if rec["allele"]]
        cons_allele = Counter(alleles).most_common(1)[0][0] if alleles else held["allele"]
        dirs = [rec["direction"] for rec in recs if rec["direction"] is not None]
        cons_dir = Counter(dirs).most_common(1)[0][0] if dirs else None

        if held["p_value"] < GW_SIG and consistent(held["allele"], held["direction"], cons_allele, cons_dir):
            label, ntype = 1, None
        elif held["p_value"] >= GW_SIG:
            label, ntype = 0, "sub_threshold"
        else:
            label, ntype = 0, "direction_inconsistent"
        candidates.append(
            {
                "snp": snp,
                "trait": trait,
                "gene": gene,
                "efo_parent": held["efo_parent"],
                "label": label,
                "negative_type": ntype,
                "cons_allele": cons_allele,
                "cons_dir": cons_dir,
                "held": held,
            }
        )
    n_sup = sum(c["label"] == 1 for c in candidates)
    print(
        f"  candidates: {len(candidates)} (supported {n_sup} | not {len(candidates) - n_sup}); "
        f"P1 dropped {n_drop_mhc} MHC + {n_drop_repl} over-replicated targets"
    )

    # ---- evidence pool: gather the related (A-E) stubs, most-relevant-first, deduped (P3/P5) ----
    def gather_related(cand):
        snp, trait, gene = cand["snp"], cand["trait"], cand["gene"]
        parent = cand["efo_parent"]
        tpos = cand["held"]["pos"]

        def not_target_pair(rec):
            return not (rec["snp"] == snp and rec["trait"] == trait)  # P5

        # A: different SNP, same gene, same trait
        tier_a = [
            stub(rec, "same_gene_same_trait", True, tpos) for rec in by_gene_trait[(gene, trait)] if rec["snp"] != snp
        ]
        # B: different SNP, same gene, trait in same EFO parent (excl. target trait)
        tier_b = [
            stub(rec, "same_gene_related_trait", True, tpos)
            for rec in by_gene[gene]
            if rec["snp"] != snp and rec["trait"] != trait and rec["efo_parent"] == parent
        ]
        # C: same SNP, different trait, same EFO parent
        tier_c = [
            stub(rec, "same_snp_related_trait", True)
            for rec in by_snp[snp]
            if rec["trait"] != trait and rec["efo_parent"] == parent
        ]
        # D: same SNP, trait in a different EFO parent
        tier_d = [
            stub(rec, "same_snp_unrelated_trait", True)
            for rec in by_snp[snp]
            if rec["trait"] != trait and rec["efo_parent"] != parent
        ]
        # E: SNP in a Reactome pathway-neighbor gene, associated with the target trait
        tier_e = []
        for ng in pathway_neighbors(gene):
            for rec in by_gene_trait[(ng, trait)]:
                if not_target_pair(rec):
                    tier_e.append(stub(rec, "pathway_gene_same_trait", True, tpos))
        # dedup, most-relevant-first, dropping any leakage of the target pair (P5). NOTE: same-gene /
        # pathway stubs on a *different* SNP in high LD with the target SNP are effectively the same
        # signal --- a subtle residual leakage this exact-pair check does not catch; filtering it needs
        # an LD reference panel and is a documented limitation (see the module docstring).
        seen, related = set(), []
        for s in tier_a[:6] + tier_c[:3] + tier_b[:8] + tier_d[:4] + tier_e[:6]:
            key = (s["snp"], s["trait"], s["study_metadata"]["accession"])
            if not (s["snp"] == snp and s["trait"] == trait) and key not in seen:
                seen.add(key)
                related.append(s)
        return related

    # ---- assemble a pool at a target sufficiency tier (P4): control how many *informative*
    # (genome-wide-significant) related stubs go in --- rich >=4, moderate 1-3, poor 0 --- then fill
    # with suggestive related stubs and unrelated controls to pool_size. Because the GWAS Catalog is a
    # catalog of hits, related stubs skew significant, so "poor" pools drop the significant related
    # evidence and lean on suggestive related + controls (the calibrated response there is the prior).
    def assemble_pool(cand, related_all, target_tier):
        gene, trait, parent = cand["gene"], cand["trait"], cand["efo_parent"]
        n_rel = round(args.pool_size * args.relevant_fraction)
        n_sig = {"rich": n_rel, "moderate": rng.randint(1, 3), "poor": 0}[target_tier]

        seen = {(s["snp"], s["trait"], s["study_metadata"]["accession"]) for s in related_all}
        related, sig_used = [], 0
        for s in related_all:  # preserve relevance order; cap the informative stubs to hit the tier
            if len(related) >= n_rel:
                break
            if informative(s):
                if sig_used >= n_sig:
                    continue
                sig_used += 1
            related.append(s)

        controls, tries = [], 0
        while len(related) + len(controls) < args.pool_size and tries < 400:
            tries += 1
            cg = rng.choice(all_genes)
            if cg == gene:
                continue
            rec = rng.choice(by_gene[cg])
            if rec["gene"] == gene or rec["trait"] == trait or rec["efo_parent"] == parent:
                continue
            key = (rec["snp"], rec["trait"], rec["study"])
            if key in seen:
                continue
            seen.add(key)
            controls.append(stub(rec, "control", False))
        pool = related + controls
        rng.shuffle(pool)
        return pool[: args.pool_size]

    # ---- balance to --frac-supported, cap per gene, ood tags, build rows ----
    sup = [c for c in candidates if c["label"] == 1]
    neg_sub = [c for c in candidates if c["negative_type"] == "sub_threshold"]
    neg_dir = [c for c in candidates if c["negative_type"] == "direction_inconsistent"]
    rng.shuffle(sup)
    rng.shuffle(neg_sub)
    rng.shuffle(neg_dir)

    def take(pool, k):
        out, per_gene, per_trait = [], defaultdict(int), defaultdict(int)
        for c in pool:
            if per_gene[c["gene"]] >= args.max_per_gene or per_trait[c["trait"]] >= args.max_per_trait:
                continue  # P2: no single gene or trait dominates
            per_gene[c["gene"]] += 1
            per_trait[c["trait"]] += 1
            out.append(c)
            if len(out) >= k:
                break
        return out

    n = args.n_episodes
    n_want_sup = int(n * args.frac_supported)
    n_want_neg = n - n_want_sup
    n_want_dir = int(n_want_neg * args.frac_direction_inconsistent_of_neg)
    chosen = (
        take(sup, n_want_sup) + take(neg_dir, n_want_dir) + take(neg_sub, n_want_neg - min(n_want_dir, len(neg_dir)))
    )
    rng.shuffle(chosen)
    if not chosen:
        raise ValueError("no episodes selected — too few multi-study SNP-trait pairs in the source")

    # ood_group tags (NOT the split): mark some EFO parents and a fraction of genes.
    all_parents = sorted({c["efo_parent"] for c in chosen})
    holdout_parents = set(rng.sample(all_parents, min(args.holdout_traits, len(all_parents))))
    chosen_genes = sorted({c["gene"] for c in chosen})
    holdout_genes = set(rng.sample(chosen_genes, int(len(chosen_genes) * args.holdout_gene_frac)))

    # P4 (sufficiency diversity): assign each episode a target tier (rich/moderate/poor) in ~equal
    # thirds, respecting each target's capacity for informative (genome-wide-significant) related stubs
    # --- rich needs >=4 available, moderate >=1; poor is always achievable by dropping significant
    # related evidence. Sort by capacity so the richest neighborhoods become the rich pools.
    prepared = [(c, gather_related(c)) for c in chosen]
    for c, rel in prepared:
        c["_cap_sig"] = sum(1 for s in rel if informative(s))
    prepared.sort(key=lambda t: -t[0]["_cap_sig"])
    m, third = len(prepared), len(prepared) // 3
    episodes = []
    for i, (c, rel) in enumerate(prepared):
        cap = c["_cap_sig"]
        if i < third:
            tier = "rich" if cap >= 4 else "moderate" if cap >= 1 else "poor"
        elif i >= m - third:
            tier = "poor"
        else:
            tier = "moderate" if cap >= 1 else "poor"
        episodes.append((c, rel, tier))
    rng.shuffle(episodes)  # undo the capacity sort so split/order isn't correlated with sufficiency

    print(f"building {len(episodes)} episodes ...")
    rows = []
    for c, rel, target_tier in episodes:
        held = c["held"]
        parent = c["efo_parent"]
        pool = assemble_pool(c, rel, target_tier)
        nconc = sum(informative(s) for s in pool if s["related"])
        tier = "rich" if nconc >= 4 else "moderate" if nconc >= 1 else "poor"  # achieved tier
        ood_group = "trait" if parent in holdout_parents else "gene" if c["gene"] in holdout_genes else "iid"
        rows.append(
            {
                "episode_id": str(uuid.uuid4())[:12],
                "target_snp": c["snp"],
                "target_trait": c["trait"],
                "target_gene": c["gene"],
                "target_allele": c["cons_allele"],
                "target_direction": c["cons_dir"],  # +1 / -1 / None
                "target_efo_parent": parent,
                "label": int(c["label"]),
                "negative_type": c["negative_type"],
                "sufficiency_tier": tier,
                "ood_group": ood_group,
                "ground_truth": {  # the held-out replication record that DEFINES the label
                    "p_value": held["p_value"],
                    "or_beta": held["or_beta"],
                    "allele": held["allele"],
                    "direction": held["direction"],
                    "study": held["study"],
                },
                "target_snp_meta": {
                    "rsid": c["snp"],
                    "chr": held["chr"],
                    "pos": held["pos"],
                    "risk_allele": c["cons_allele"],
                    "raf": held["raf"],
                    "functional_class": held["functional_class"],
                    "intergenic": held["intergenic"],
                },
                "target_gene_meta": gene_meta(c["gene"], held["gene_ensembl"]),
                "target_trait_meta": {"trait": c["trait"], "efo_parent": parent},
                "context_pool": pool,
            }
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # stratified-iid splits: held-out test + small val for monitoring, rest train. The split is the
    # FILE (no `split` column) per the shared loader convention; ood_group stays a metadata column.
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

    # composition stats (also printed in QC below) --- recorded so the dataset self-documents.
    mean_related_fraction = float(
        df["context_pool"].map(lambda p: np.mean([s["related"] for s in p]) if len(p) else 0.0).mean()
    )
    tier_fractions = (
        pd.Series([s["context_type"] for p in df["context_pool"] for s in p])
        .value_counts(normalize=True)
        .round(4)
        .to_dict()
    )
    n_neg_cand = len(neg_sub) + len(neg_dir)
    manifest = {
        "version": "v1",
        "n_records": len(df),
        "seed": args.seed,
        "n_snps": int(df["target_snp"].nunique()),
        "n_genes": int(df["target_gene"].nunique()),
        "n_traits": int(df["target_trait"].nunique()),
        "n_efo_parents": int(df["target_efo_parent"].nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
        "negative_type_counts": df["negative_type"].value_counts(dropna=False).to_dict(),
        "sufficiency_counts": df["sufficiency_tier"].value_counts().to_dict(),
        "ood_group_counts": df["ood_group"].value_counts().to_dict(),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "pool_size": args.pool_size,
        "mean_pool_len": float(df["context_pool"].map(len).mean()),
        # candidate pool + the natural-negative-bounded balanced ceiling (positives are abundant).
        "n_candidates": len(candidates),
        "candidate_class_counts": {
            "supported": len(sup),
            "sub_threshold": len(neg_sub),
            "direction_inconsistent": len(neg_dir),
        },
        "balanced_ceiling": 2 * min(len(sup), n_neg_cand),
        # KNOWN CHARACTERISTIC: related:control runs below --relevant-fraction because the Catalog is a
        # catalog of genome-wide-significant hits, so non-significant related evidence is scarce and the
        # P4 'poor' pools (~1/3 of episodes) fill largely with controls.
        "mean_related_fraction": round(mean_related_fraction, 4),
        "target_related_fraction": args.relevant_fraction,
        "stub_tier_fractions": tier_fractions,
        "params": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ---- QC ----
    print(f"\nwrote train/val/test ({len(train_df)}/{len(val_df)}/{len(test_df)}) + manifest.json -> {out}")
    print(f"label: supported {100 * (df.label == 1).mean():.1f}% | not {100 * (df.label == 0).mean():.1f}%")
    print("negatives split: " + str(df[df.label == 0].negative_type.value_counts(normalize=True).round(3).to_dict()))
    print("sufficiency: " + str(df.sufficiency_tier.value_counts(normalize=True).round(3).to_dict()))
    print("ood_group: " + str(df.ood_group.value_counts().to_dict()))
    print(f"distinct target genes: {df.target_gene.nunique()} | traits: {df.target_trait.nunique()}")
    poolsz = df.context_pool.map(len)
    relfrac = df.context_pool.map(lambda p: np.mean([s["related"] for s in p]) if len(p) else 0.0)
    print(
        f"pool size: mean {poolsz.mean():.1f} | related fraction: mean {relfrac.mean():.2f} (target {args.relevant_fraction})"
    )
    tiers = pd.Series([s["context_type"] for p in df.context_pool for s in p]).value_counts(normalize=True).round(3)
    print(f"stub tiers: {tiers.to_dict()}")
    # leakage audit (P5): the target (snp, trait) must never appear as a stub in its own pool
    leak = sum(
        1 for r in rows for s in r["context_pool"] if s["snp"] == r["target_snp"] and s["trait"] == r["target_trait"]
    )
    print(f"P5 leakage (target SNP-trait in own pool): {leak} (must be 0)")


if __name__ == "__main__":
    main()
