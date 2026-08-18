#!/usr/bin/env python3
"""
perte_info_analyse.py

Analyse de la perte d'information (couverture inter-patients) sur la matrice
cohorte de méthylation (sites x patients), projet STING.

Structure attendue de matrice_cohorte.tsv (séparateur tabulation) :
  - colonnes méta fixes : site_id, chrom, start, mod_code, origin,
    feature_label, gene_name, distance_TSS, ccre_accession, ccre_type
  - colonnes patients appariées : beta_<patient_id> / cov_<patient_id>
    (beta = valeur de méthylation, NaN si non couvert ;
     cov  = profondeur de lecture réelle au site, pour ce patient)

Un site est considéré "couvert" par un patient si beta_<patient_id> est
non-NaN (mesure de méthylation valide).

Produit :
  1. coverage_summary.csv   : pour chaque nombre de patients x couvrant
                               EXACTEMENT un site (x = 1..N), statistiques
                               agrégées (n_sites, n_nan_values,
                               n_sites_with_nan, et une colonne
                               n_origin_<valeur> par catégorie observée
                               dans la colonne 'origin').
  2. per_site_coverage.csv  : les 5000 sites les MOINS bien couverts
                               (site_id, chrom, start, cov_<patient> pour
                               chaque patient, n_covered).

Usage:
  python perte_info_analyse.py \
      --input matrice_cohorte.tsv \
      --outdir results/coverage_loss \
      --n-least-covered 5000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHROM_ORDER = [str(i) for i in range(1, 23)] + ["X", "Y", "M"]

FIXED_META_COLS = [
    "site_id", "chrom", "start", "mod_code", "origin",
    "feature_label", "gene_name", "distance_TSS",
    "ccre_accession", "ccre_type",
]


def normalize_chrom_label(chrom: str) -> str:
    c = chrom.lower().replace("chr", "")
    return c.upper() if c in {"x", "y", "m"} else c


def build_chrom_sort_key(chrom_values: list[str]) -> dict[str, int]:
    present = {normalize_chrom_label(c): c for c in chrom_values}
    ordered = [present[c] for c in CHROM_ORDER if c in present]
    leftovers = sorted(set(chrom_values) - set(ordered))
    if leftovers:
        log.warning(
            "Chromosomes hors liste standard détectés et placés en fin de "
            "tri : %s", leftovers,
        )
    full_order = ordered + leftovers
    return {c: i for i, c in enumerate(full_order)}


def detect_patient_pairs(columns: list[str]) -> list[tuple[str, str, str]]:
    """Détecte les paires (patient_id, beta_col, cov_col) à partir des
    colonnes beta_<id> / cov_<id>. Assertion défensive : chaque beta_ doit
    avoir son cov_ correspondant, et inversement, sinon échec explicite
    plutôt qu'un patient silencieusement ignoré ou mal apparié."""
    beta_cols = {c: c[len("beta_"):] for c in columns if c.startswith("beta_")}
    cov_cols = {c: c[len("cov_"):] for c in columns if c.startswith("cov_")}

    beta_ids = set(beta_cols.values())
    cov_ids = set(cov_cols.values())

    missing_cov = beta_ids - cov_ids
    missing_beta = cov_ids - beta_ids
    if missing_cov:
        raise ValueError(
            f"Colonnes beta_ sans colonne cov_ correspondante : {sorted(missing_cov)}"
        )
    if missing_beta:
        raise ValueError(
            f"Colonnes cov_ sans colonne beta_ correspondante : {sorted(missing_beta)}"
        )

    pairs = sorted(
        [(pid, f"beta_{pid}", f"cov_{pid}") for pid in beta_ids],
        key=lambda t: t[0],
    )
    return pairs


def load_matrix(path: Path) -> tuple[pl.DataFrame, list[tuple[str, str, str]]]:
    log.info("Lecture de %s (Polars)...", path)
    # "NaN"/"NA" textuels (écriture NumPy/pandas typique) doivent être
    # explicitement déclarés : Polars ne reconnaît par défaut que la chaîne
    # vide comme null, sinon la couverture serait silencieusement surestimée.
    lf = pl.scan_csv(
        path, separator="\t", infer_schema_length=10000,
        null_values=["NaN", "nan", "NA", "N/A", ""],
    )
    columns = lf.collect_schema().names()

    missing_meta = [c for c in FIXED_META_COLS if c not in columns]
    if missing_meta:
        raise ValueError(f"Colonnes méta absentes du fichier : {missing_meta}")

    pairs = detect_patient_pairs(columns)
    log.info("%d patients détectés (paires beta_/cov_).", len(pairs))

    df = lf.collect()
    return df, pairs


def compute_coverage(df: pl.DataFrame, pairs: list[tuple[str, str, str]]) -> pl.DataFrame:
    beta_cols = [b for _, b, _ in pairs]
    log.info("Calcul de n_covered (beta non-NaN) sur %d patients...", len(pairs))
    df = df.with_columns(
        pl.sum_horizontal(
            [pl.col(b).is_not_null().cast(pl.Int32) for b in beta_cols]
        ).alias("n_covered")
    )
    return df


def write_per_site_coverage(
    df: pl.DataFrame,
    pairs: list[tuple[str, str, str]],
    n_least_covered: int,
    output_path: Path,
) -> None:
    log.info(
        "Sélection des %d sites les moins couverts pour per_site_coverage.csv...",
        n_least_covered,
    )
    cov_cols = [c for _, _, c in pairs]

    out = (
        df.select(["site_id", "chrom", "start", "n_covered", *cov_cols])
        .sort("n_covered", descending=False)
        .head(n_least_covered)
    )
    out.write_csv(output_path)
    log.info(
        "-> %s (%d lignes, n_covered de %d à %d)",
        output_path, out.height,
        out["n_covered"].min(), out["n_covered"].max(),
    )


def write_covered_sites_bed(
    df: pl.DataFrame,
    min_coverage_pct: float,
    output_path: Path,
) -> None:
    """Write a BED file with sites covered in at least X% of the cohort.

    The BED columns are: chrom, start, end, site_id.
    The file name follows the convention:
      coverage_distribution_covered_at_least_{pct}pct_before_imputation.bed
    """
    if not 0 <= min_coverage_pct <= 100:
        raise ValueError(f"min_coverage_pct must be between 0 and 100, got {min_coverage_pct}")

    n_total = df["n_covered"].len() if df.height else 0
    if n_total == 0:
        log.warning("Aucun site trouvé ; BED vide écrite dans %s", output_path)
        pl.DataFrame({"chrom": [], "start": [], "end": [], "site_id": []}).write_csv(
            output_path,
            separator="\t",
            include_header=False,
        )
        return

    # Keep only sites with coverage >= pct of the cohort.
    # 'n_covered' is already the number of non-null beta values among patients.
    # The cohort size is total number of patients, not number of sites.
    # A site is retained if n_covered / n_patients >= pct/100.
    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    n_patients = len(beta_cols)
    if n_patients == 0:
        raise ValueError("Aucune colonne beta_<patient> détectée dans la matrice : impossible de calculer le seuil de couverture")

    keep = (
        df.filter(pl.col("n_covered") / n_patients * 100 >= min_coverage_pct)
        .with_columns([
            pl.col("chrom").cast(pl.Utf8),
            pl.col("start").cast(pl.Int64),
            (pl.col("start") + 2).alias("end"),
            pl.col("site_id").cast(pl.Utf8),
        ])
        .select(["chrom", "start", "end", "site_id"])
        .sort(["chrom", "start"])
    )

    keep.write_csv(output_path, separator="\t", include_header=False)
    log.info(
        "-> %s (%d sites couverts au moins à %.2f%% de la cohorte)",
        output_path, keep.height, min_coverage_pct,
    )


def write_coverage_summary(
    df: pl.DataFrame,
    n_total: int,
    origin_col: str,
    output_path: Path,
) -> None:
    log.info("Agrégation par nombre exact de patients couvrants (x = 1..%d)...", n_total)

    n_sites_x0 = df.filter(pl.col("n_covered") == 0).height
    if n_sites_x0 > 0:
        log.warning(
            "%d sites ont une couverture nulle (x=0) : exclus du CSV1 "
            "conformément à la spécification (x parcourt 1..N).", n_sites_x0,
        )

    # origin peut être une concaténation de tags séparés par ';'
    # (ex. "450k;cpgIsland"). Un site est compté UNE SEULE FOIS dans n_EPC
    # s'il porte au moins un des tags 450k/850k/v2 (union, pas somme des
    # trois -> évite le double comptage d'un site multi-tag EPIC).
    EPC_TAGS = {"450k", "850k", "v2"}
    df_pos = df.filter(pl.col("n_covered") > 0).with_columns(
        pl.col(origin_col).fill_null("").str.split(";").alias("_origin_tags")
    )
    df_pos = df_pos.with_columns(
        [
            pl.col("_origin_tags")
            .list.eval(pl.element().is_in(list(EPC_TAGS)))
            .list.any()
            .alias("is_epc"),
            pl.col("_origin_tags")
            .list.eval(pl.element() == "cpgIsland")
            .list.any()
            .alias("is_island"),
        ]
    )

    base = (
        df_pos.group_by("n_covered")
        .agg(
            [
                pl.len().alias("n_sites"),
                pl.col("is_epc").sum().alias("n_EPC"),
                pl.col("is_island").sum().alias("n_islandcpg"),
            ]
        )
        .sort("n_covered")
    )
    full_x = pl.DataFrame({"n_covered": list(range(1, n_total + 1))})
    base = full_x.join(base, on="n_covered", how="left").fill_null(0)
    base = base.with_columns(
        [
            (pl.col("n_sites") * (n_total - pl.col("n_covered"))).alias("n_nan_values"),
            pl.when(pl.col("n_covered") < n_total)
            .then(pl.col("n_sites"))
            .otherwise(0)
            .alias("n_sites_with_nan"),
        ]
    )

    agg = base.sort("n_covered").rename({"n_covered": "x_patients_couvrants"}).select(
        [
            "x_patients_couvrants",
            "n_sites",
            "n_sites_with_nan",
            "n_nan_values",
            "n_EPC",
            "n_islandcpg",
        ]
    )

    # Assertions défensives : cohérence des totaux (x=0 exclus, cf. warning).
    total_check = agg["n_sites"].sum() + n_sites_x0
    assert total_check == df.height, (
        f"Incohérence : somme des n_sites par groupe ({total_check}) != "
        f"nombre total de sites ({df.height})"
    )
    expected_epc = df_pos.filter(pl.col("is_epc")).height
    expected_island = df_pos.filter(pl.col("is_island")).height
    assert agg["n_EPC"].sum() == expected_epc, (
        f"Incohérence n_EPC : {agg['n_EPC'].sum()} != {expected_epc}"
    )
    assert agg["n_islandcpg"].sum() == expected_island, (
        f"Incohérence n_islandcpg : {agg['n_islandcpg'].sum()} != {expected_island}"
    )

    agg.write_csv(output_path)
    log.info("-> %s (%d lignes, x=1..%d)", output_path, agg.height, n_total)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Matrice cohorte TSV (sites x patients)")
    parser.add_argument("--outdir", required=True, type=Path,
                        help="Répertoire de sortie pour les CSV et figure")
    parser.add_argument("--origin-col", default="origin",
                        help="Nom de la colonne d'origine dans la matrice (défaut: 'origin')")
    parser.add_argument(
        "--n-least-covered", type=int, default=5000,
        help="Nombre de sites les moins couverts à inclure dans per_site_coverage.csv",
    )
    parser.add_argument(
        "--min-coverage-pct", type=float, default=75.0,
        help="Seuil de couverture minimum exprimé en % de la cohorte pour exporter le BED de sites couverts (défaut: 75)",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df, pairs = load_matrix(args.input)
    n_total = len(pairs)

    df = compute_coverage(df, pairs)

    write_per_site_coverage(
        df, pairs, args.n_least_covered,
        args.outdir / "per_site_coverage.csv",
    )
    write_coverage_summary(
        df, n_total, args.origin_col,
        args.outdir / "coverage_summary.csv",
    )

    bed_output = args.outdir / (
        f"coverage_distribution_covered_at_least_{args.min_coverage_pct:g}pct_before_imputation.bed"
    )
    write_covered_sites_bed(df, args.min_coverage_pct, bed_output)

    log.info("Terminé. N_patients = %d, N_sites = %d", n_total, df.height)


if __name__ == "__main__":
    main()