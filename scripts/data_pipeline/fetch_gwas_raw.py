"""Download the raw inputs for build_gwas_dataset.py into a local dir.

Pulls the sources the builder reads, into <out-dir> with the exact filenames it expects:

  gwas_catalog_associations.tsv   NHGRI-EBI GWAS Catalog, all associations w/ ontology annotations
  gwas_trait_mappings.tsv         GWAS Catalog trait_mappings; its 'Parent term' column IS the EFO
                                  parent category — a REQUIRED build input
  Ensembl2Reactome.txt            Reactome Ensembl-gene -> pathway map (enables tier E + enrichment)
  gtex_gene_median_tpm.gct.gz     GTEx v8 median TPM per gene (tissue-expression enrichment)

EFO parent categories are taken directly from trait_mappings' 'Parent term' column — no ontology
traversal or hand-derived file — so the whole build is reproducible from these downloads alone.

    .venv/bin/python scripts/data_pipeline/fetch_gwas_raw.py \\
        --out-dir ibis_data/raw/gwas/raw
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

# Sources -> (url, is_zip). The GWAS Catalog associations + trait-mappings live on the EBI FTP (the
# old /gwas/api/search/downloads/ endpoints are gone). The associations file ships as a zip holding a
# single ontology-annotated TSV (columns MAPPED_TRAIT / MAPPED_TRAIT_URI / SNP_GENE_IDS, ... that the
# builder reads); gwas-efo-trait-mappings.tsv carries the 'Parent term' column used for EFO parents.
SOURCES = {
    "gwas_catalog_associations.tsv": (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/"
        "gwas-catalog-associations_ontology-annotated-full.zip",
        True,
    ),
    "gwas_trait_mappings.tsv": (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-efo-trait-mappings.tsv",
        False,
    ),
    "Ensembl2Reactome.txt": ("https://reactome.org/download/current/Ensembl2Reactome.txt", False),
    "gtex_gene_median_tpm.gct.gz": (
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
        "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
        False,
    ),
}


def _download(url: str, dest: Path, is_zip: bool) -> None:
    print(f"  {url}\n    -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "ibis-gwas-fetch/1.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted hosts)
        data = resp.read()
    if is_zip:  # extract the single TSV inside the associations archive
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            inner = next(n for n in z.namelist() if n.endswith((".tsv", ".txt", ".csv")))
            dest.write_bytes(z.read(inner))
    else:
        dest.write_bytes(data)
    print(f"    done ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ibis_data/raw/gwas/raw")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"fetching GWAS raw inputs -> {out}")
    for fname, (url, is_zip) in SOURCES.items():
        dest = out / fname
        if dest.exists() and not args.force:
            print(f"  skip {fname} (exists; --force to re-download)")
            continue
        _download(url, dest, is_zip)

    print("\ndone. All build inputs fetched; EFO parents come from trait_mappings' 'Parent term' column.")


if __name__ == "__main__":
    main()
