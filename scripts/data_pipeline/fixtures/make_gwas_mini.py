"""Generate the committed `gwas_mini` fixture for the hermetic GWAS build test.

Unlike `make_depmap_mini.py` (which slices multi-GB real CSVs), the GWAS Catalog association
format is small and self-contained, so this synthesizes a tiny, internally-consistent world
directly — no real source needed. The output is a schema-faithful GWAS Catalog associations TSV
(the exact columns the builder reads) plus tiny Reactome / GTEx / trait_mappings side files, wired so
`build_gwas_dataset.py` runs end-to-end in seconds and every relevance tier + label class is
exercised:

  - genes grouped into shared Reactome pathways (so tier E pathway-neighbors exist),
  - traits grouped under EFO parents (so tiers B/C "same parent" and D "different parent" exist),
  - each SNP-trait hypothesis gets >=2 study records with a controlled held-out (largest-sample)
    record producing a supported / sub_threshold / direction_inconsistent label in round-robin,
  - a handful of multi-gene SNPs + multi-SNP haplotype rows the builder must filter out.

Run once; the output under `gwas_mini/` is committed.

    .venv/bin/python scripts/data_pipeline/fixtures/make_gwas_mini.py
"""

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np

# GWAS Catalog association columns the builder reads (a faithful subset of the real file).
COLUMNS = [
    "SNPS",
    "MAPPED_TRAIT",
    "MAPPED_TRAIT_URI",
    "SNP_GENE_IDS",
    "MAPPED_GENE",
    "CHR_ID",
    "CHR_POS",
    "STRONGEST SNP-RISK ALLELE",
    "RISK ALLELE FREQUENCY",
    "CONTEXT",
    "INTERGENIC",
    "STUDY ACCESSION",
    "INITIAL SAMPLE SIZE",
    "PLATFORM [SNPS PASSING QC]",
    "GENOTYPING TECHNOLOGY",
    "P-VALUE",
    "OR or BETA",
    "95% CI (TEXT)",
]
PARENTS = [
    "Metabolic disorder",
    "Cardiovascular disease",
    "Cancer",
    "Immune system disorder",
    "Body measurement",
    "Neurological disorder",
]
ANCESTRIES = ["European", "East Asian", "South Asian", "African American", "Hispanic"]
FUNC = ["intron_variant", "missense_variant", "intergenic_variant", "regulatory_region_variant"]
CLASSES = ["sup", "sub", "dir"]  # supported / sub-threshold / direction-inconsistent, round-robin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "gwas_mini"))
    ap.add_argument("--n-genes", type=int, default=48)
    ap.add_argument("--genes-per-pathway", type=int, default=4)
    ap.add_argument("--snps-per-gene", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- world: genes -> pathway + home parent; traits per parent ----
    genes = [f"GN{i:03d}" for i in range(args.n_genes)]
    ens_of = {g: f"ENSG{i:011d}" for i, g in enumerate(genes)}
    pathway_of = {g: f"R-HSA-{1000 + (i // args.genes_per_pathway)}" for i, g in enumerate(genes)}
    # genes in the same pathway share a home parent (so tier-E neighbors share the target trait's parent)
    parent_of_gene = {g: PARENTS[(i // args.genes_per_pathway) % len(PARENTS)] for i, g in enumerate(genes)}
    traits_of_parent = {p: [f"{p.lower().replace(' ', '_')}_trait_{k}" for k in range(4)] for p in PARENTS}
    trait_uri = {}
    n = 0
    for p in PARENTS:
        for t in traits_of_parent[p]:
            trait_uri[t] = f"http://www.ebi.ac.uk/efo/EFO_{2000000 + n}"
            n += 1
    parent_of_trait = {t: p for p in PARENTS for t in traits_of_parent[p]}
    other_parent = {p: PARENTS[(i + 1) % len(PARENTS)] for i, p in enumerate(PARENTS)}

    # study pool: accession -> baseline sample size (held-out record gets a boosted size, below)
    studies = [(f"GCST{90000 + i:06d}", int(rng.integers(20_000, 120_000))) for i in range(60)]

    rows = []
    snp_counter = 0
    class_i = 0

    def emit_pair(snp, trait, gene, allele, cls, ancestry, n_studies=3):
        """Emit >=2 study records for one (snp, trait) hypothesis with a controlled held-out label."""
        chosen = [studies[i] for i in rng.choice(len(studies), size=n_studies, replace=False)]
        chr_id = str(1 + (hash(gene) % 22))
        pos = 1_000_000 + (snp_counter * 137 % 5_000_000)
        raf = round(float(rng.uniform(0.05, 0.6)), 3)
        # The builder's held-out ground truth under model (A) is the MOST-SIGNIFICANT record (min p),
        # and the hypothesis direction is the modal call, so:
        #   sup: every record genome-wide significant + increases (min p is significant, consistent)
        #   sub: every record only suggestive (p >= 5e-8) -> min p >= 5e-8 -> sub_threshold negative
        #   dir: majority increase at moderate p, but the single most-significant record flips to
        #        decrease -> min p is direction-inconsistent with the modal hypothesis
        for k, (acc, base_n) in enumerate(chosen):
            samp_n = base_n + snp_counter
            p_value, ra, orbeta, ci = 1e-12, allele, round(float(rng.uniform(1.1, 1.6)), 3), ""
            if cls == "sub":
                p_value = float(rng.uniform(6e-8, 8e-7))  # no genome-wide hit for the whole pair
            elif cls == "dir":
                if k == 0:  # the most-significant record flips direction (OR < 1)
                    p_value, orbeta = 1e-20, round(float(rng.uniform(0.6, 0.9)), 3)
                else:  # majority say increase, but at weaker significance
                    p_value, orbeta = 1e-9, round(float(rng.uniform(1.1, 1.6)), 3)
            rows.append(
                {
                    "SNPS": snp,
                    "MAPPED_TRAIT": trait,
                    "MAPPED_TRAIT_URI": trait_uri[trait],
                    "SNP_GENE_IDS": ens_of[gene],
                    "MAPPED_GENE": gene,
                    "CHR_ID": chr_id,
                    "CHR_POS": str(pos),
                    "STRONGEST SNP-RISK ALLELE": f"{snp}-{ra}",
                    "RISK ALLELE FREQUENCY": str(raf),
                    "CONTEXT": FUNC[snp_counter % len(FUNC)],
                    "INTERGENIC": "0",
                    "STUDY ACCESSION": acc,
                    "INITIAL SAMPLE SIZE": f"{samp_n:,} {ancestry} individuals",
                    "PLATFORM [SNPS PASSING QC]": "Illumina",
                    "GENOTYPING TECHNOLOGY": "Genome-wide genotyping array",
                    "P-VALUE": f"{p_value:.1e}",
                    "OR or BETA": str(orbeta),
                    "95% CI (TEXT)": ci,
                }
            )

    for gi, gene in enumerate(genes):
        parent = parent_of_gene[gene]
        home = traits_of_parent[parent][0]
        related = traits_of_parent[parent][1]  # same parent, different trait (tiers B/C)
        unrelated = traits_of_parent[other_parent[parent]][0]  # different parent (tier D)
        allele = "ACGT"[gi % 4]
        anc = ANCESTRIES[gi % len(ANCESTRIES)]
        gene_snps = []
        for s in range(args.snps_per_gene):
            snp = f"rs{1_000_000 + snp_counter}"
            gene_snps.append(snp)
            snp_counter += 1
        # every SNP -> home trait (tier A: different SNPs, same gene, same trait; all are candidates)
        for snp in gene_snps:
            emit_pair(snp, home, gene, allele, CLASSES[class_i % len(CLASSES)], anc)
            class_i += 1
        # snp[1] -> related trait (tier B: different SNP, same gene, related trait)
        emit_pair(gene_snps[1], related, gene, allele, CLASSES[class_i % len(CLASSES)], anc)
        class_i += 1
        # snp[0] -> related trait (tier C: same SNP, related trait) and unrelated (tier D)
        emit_pair(gene_snps[0], related, gene, allele, CLASSES[class_i % len(CLASSES)], anc)
        class_i += 1
        emit_pair(gene_snps[0], unrelated, gene, allele, CLASSES[class_i % len(CLASSES)], anc)
        class_i += 1

    # a few rows the builder must FILTER OUT: multi-gene SNPs and multi-SNP haplotypes
    rows.append(
        {c: "" for c in COLUMNS}
        | {
            "SNPS": "rs9999001",
            "MAPPED_TRAIT": "metabolic_disorder_trait_0",
            "MAPPED_TRAIT_URI": trait_uri["metabolic_disorder_trait_0"],
            "SNP_GENE_IDS": f"{ens_of['GN000']}, {ens_of['GN001']}",  # multi-gene -> dropped
            "MAPPED_GENE": "GN000, GN001",
            "CHR_ID": "1",
            "CHR_POS": "50000",
            "STRONGEST SNP-RISK ALLELE": "rs9999001-A",
            "STUDY ACCESSION": "GCST099999",
            "INITIAL SAMPLE SIZE": "10,000 European individuals",
            "P-VALUE": "1e-9",
            "OR or BETA": "1.2",
            "95% CI (TEXT)": "",
        }
    )
    rows.append(
        {c: "" for c in COLUMNS}
        | {
            "SNPS": "rs9999002; rs9999003",  # haplotype -> dropped
            "MAPPED_TRAIT": "cancer_trait_0",
            "MAPPED_TRAIT_URI": trait_uri["cancer_trait_0"],
            "SNP_GENE_IDS": ens_of["GN010"],
            "MAPPED_GENE": "GN010",
            "CHR_ID": "3",
            "CHR_POS": "70000",
            "STRONGEST SNP-RISK ALLELE": "rs9999002-?",
            "STUDY ACCESSION": "GCST099998",
            "INITIAL SAMPLE SIZE": "8,000 European individuals",
            "P-VALUE": "1e-9",
            "OR or BETA": "1.1",
            "95% CI (TEXT)": "",
        }
    )
    # an MHC/HLA-region pair (chr6, single gene, genome-wide significant): a valid pair that would be a
    # supported candidate but for P1 --- it must be EXCLUDED as a target, though still usable as evidence.
    for acc in ("GCST088801", "GCST088802"):
        rows.append(
            {c: "" for c in COLUMNS}
            | {
                "SNPS": "rs8000001",
                "MAPPED_TRAIT": "immune_system_disorder_trait_0",
                "MAPPED_TRAIT_URI": trait_uri["immune_system_disorder_trait_0"],
                "SNP_GENE_IDS": "ENSG00000900001",
                "MAPPED_GENE": "HLADRB1",
                "CHR_ID": "6",
                "CHR_POS": "30000000",  # inside the extended MHC window (25-34 Mb)
                "STRONGEST SNP-RISK ALLELE": "rs8000001-A",
                "RISK ALLELE FREQUENCY": "0.3",
                "CONTEXT": "intron_variant",
                "INTERGENIC": "0",
                "STUDY ACCESSION": acc,
                "INITIAL SAMPLE SIZE": "100,000 European individuals",
                "PLATFORM [SNPS PASSING QC]": "Illumina",
                "GENOTYPING TECHNOLOGY": "Genome-wide genotyping array",
                "P-VALUE": "1e-15",
                "OR or BETA": "1.3",
                "95% CI (TEXT)": "",
            }
        )

    # ---- write associations TSV ----
    with open(out / "gwas_catalog_associations.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})

    # ---- Reactome: gene ensembl -> pathway (human) ----
    with open(out / "Ensembl2Reactome.txt", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        for g in genes:
            pid = pathway_of[g]
            w.writerow([ens_of[g], pid, f"https://reactome.org/{pid}", f"Pathway {pid}", "IEA", "Homo sapiens"])

    # ---- GTEx: median TPM per gene (versioned Ensembl in Name) ----
    tissues = ["Liver", "Whole_Blood", "Brain_Cortex", "Adipose_Subcutaneous", "Muscle_Skeletal"]
    with gzip.open(out / "gtex_gene_median_tpm.gct.gz", "wt", newline="") as fh:
        fh.write("#1.2\n")
        fh.write(f"{len(genes)}\t{len(tissues)}\n")
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["Name", "Description"] + tissues)
        for g in genes:
            vals = np.round(rng.uniform(0, 120, size=len(tissues)), 1)
            w.writerow([f"{ens_of[g]}.1", g] + list(vals))

    # ---- GWAS Catalog trait_mappings (EFO URI -> parent category), the real download's schema ----
    with open(out / "gwas_trait_mappings.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["Disease trait", "EFO term", "EFO URI", "Parent term", "Parent URI"])
        for t, p in parent_of_trait.items():
            parent_uri = f"http://www.ebi.ac.uk/efo/EFO_{9000000 + PARENTS.index(p)}"
            w.writerow([t, t, trait_uri[t], p, parent_uri])

    print(f"wrote gwas_mini fixture -> {out}")
    print(f"  associations rows: {len(rows)} | genes: {len(genes)} | traits: {len(parent_of_trait)}")


if __name__ == "__main__":
    main()
