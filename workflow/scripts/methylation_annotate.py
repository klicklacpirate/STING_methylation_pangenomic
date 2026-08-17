#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Methylation cohort pipeline — bedMethyl (Modkit) → matrice de cohorte filtrée
==============================================================================

Pipeline en 5 étapes :

  1. parse_sample
        {sample}_pileup.bed (bedMethyl Modkit) → {sample}.parquet
        - Garde uniquement mod_code 'm' (5mC)
        - site_id = "chrom:start"
        - Si le même site apparaît sur + et - : garde la moyenne de méthylation pondérée par N_valid
        - fraction remise en [0,1] si nécessaire (Modkit peut sortir 0-100)

  2. build_cohort_matrix
        Tous les {sample}.parquet → cohort_matrix.parquet
        - Outer join sur site_id
        - Seul mod_code 'm' (5mC) est traité.

  3. assign_origins
        cohort_matrix.parquet + BED source (cpgIsland / 450k / 850k / v2)
        → cohort_with_origins.parquet
        - Un site peut chevaucher plusieurs régions → origines concaténées (triées)
        - Sites sans aucun hit dans le BED source : conservés avec origine="unknown"

  4. annotate_genomic
        cohort_with_origins.parquet + BED TSS / exons / introns / cCRE
        → cohort_genomic.parquet
        - Priorité : Promoter/TSS > Exon > Intron > cCRE > Intergenic

  5. export_tsv
        cohort_genomic.parquet → methylation_cohort.tsv
        Format : une ligne par site_id (chrom:start)
        chrom, start, end, strand, mod_code, <annot...>,
        beta_<s1>, cov_<s1>, beta_<s2>, cov_<s2>, ...

──────────────────────────────────────────────────────────────────
Format bedMethyl Modkit (colonnes 0-indexed) :
  0  chrom
  1  start (0-based)
  2  end
  3  mod_code (m=5mC, h=5hmC, ...)
  4  score
  5  strand (+/-)
  6  start2
  7  end2
  8  color
  9  N_valid       ← coverage utilisé pour le filtre
  10 fraction      ← beta (0-100)
  11 N_mod
  12 N_canonical
  13 N_other_mod
  14 N_delete
  15 N_fail
  16 N_diff
  17 N_nocall

Format BED source (sans en-tête, tabulation) :
  chrom  start  end  origine    (origine ∈ { "cpgIsland", "450k", "850k", "v2" })

Format atlas (avec ou sans en-tête commençant par '#', tabulation) :
  chrom  chromStart  chromEnd  celltype  [colonnes ignorées]
──────────────────────────────────────────────────────────────────
"""

import os
from pathlib import Path
import sys
import logging
import argparse
import time

import numpy as np
import pandas as pd
import polars as pl

# ========================= LOGGING =======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ========================= HELPERS =======================

def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _build_interval_index(path: str, name_col: int = 3) -> dict:
    """
    Charge un BED (≥ name_col+1 colonnes) et construit un index NumPy par chrom.
    Retourne : { chrom : { starts, ends, names } }  trié par start.
    """
    df = pd.read_csv(
        path, sep="\t", header=None, comment="#", encoding="latin-1",
        usecols=[0, 1, 2, name_col],
        names=["chrom", "start", "end", "name"],
        dtype={0: str, 1: str, 2: str, name_col: str},
        low_memory=False,
    )
    df["start"] = pd.to_numeric(df["start"], errors="coerce").fillna(0).astype(np.int64)
    df["end"]   = pd.to_numeric(df["end"],   errors="coerce").fillna(0).astype(np.int64)
    df = df[df["start"] >= 0].copy()

    idx = {}
    for ch, grp in df.groupby("chrom", sort=False):
        g = grp.sort_values("start")
        idx[str(ch)] = {
            "starts": g["start"].to_numpy(dtype=np.int64),
            "ends":   g["end"].to_numpy(dtype=np.int64),
            "names":  g["name"].to_numpy(dtype=object),
        }
    return idx


def _vec_first_overlap(sites_df: pd.DataFrame, ref: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Pour chaque site, retourne (hit_mask, hit_name) du PREMIER intervalle
    qui le chevauche dans ref.
    Algorithme O(N log M) via searchsorted.
    """
    n = len(sites_df)
    hit_mask  = np.zeros(n, dtype=bool)
    hit_names = np.empty(n, dtype=object)
    hit_names[:] = ""

    chrom_arr = sites_df["chrom"].to_numpy(dtype=str)
    start_arr = sites_df["start"].to_numpy(dtype=np.int64)

    # Grouper les indices de sites par chrom (une seule passe)
    chrom_to_idx: dict = {}
    for i, ch in enumerate(chrom_arr):
        chrom_to_idx.setdefault(ch, []).append(i)

    for chrom, d in ref.items():
        global_idx = chrom_to_idx.get(chrom)
        if global_idx is None:
            continue
        gi = np.array(global_idx, dtype=np.int64)
        s  = start_arr[gi]

        # Candidat immédiatement à gauche (searchsorted "right" - 1)
        cand = np.searchsorted(d["starts"], s, side="right") - 1
        valid = cand >= 0
        if not valid.any():
            continue
        gi_v, ci_v, sv = gi[valid], cand[valid], s[valid]
        overlap = d["ends"][ci_v] > sv
        hit_mask[gi_v[overlap]]  = True
        hit_names[gi_v[overlap]] = d["names"][ci_v[overlap]]

    return hit_mask, hit_names


def _vec_all_overlaps(sites_df: pd.DataFrame, ref: dict) -> list[set]:
    """
    Pour chaque site, retourne l'ensemble de TOUS les noms qui le chevauchent.
    Utilisé dans assign_origins pour collecter toutes les origines.
    """
    n = len(sites_df)
    results: list[set] = [set() for _ in range(n)]

    chrom_arr = sites_df["chrom"].to_numpy(dtype=str)
    start_arr = sites_df["start"].to_numpy(dtype=np.int64)
    end_arr   = sites_df["end"].to_numpy(dtype=np.int64)

    chrom_to_idx: dict = {}
    for i, ch in enumerate(chrom_arr):
        chrom_to_idx.setdefault(ch, []).append(i)

    for chrom, d in ref.items():
        global_idx = chrom_to_idx.get(chrom)
        if global_idx is None:
            continue
        gi = np.array(global_idx, dtype=np.int64)
        s  = start_arr[gi]
        e  = end_arr[gi]

        # Pour chaque site, cherche tous les intervalles qui le chevauchent
        # Un intervalle [rs, re) chevauche [s, e) si rs < e ET re > s
        i_end = np.searchsorted(d["starts"], e, side="left")   # ref_start < site_end

        for k, (site_gi, site_s, site_e, ie) in enumerate(zip(gi, s, e, i_end)):
            if ie == 0:
                continue
            hits = np.where(d["ends"][:ie] > site_s)[0]
            for j in hits:
                # Éclater les tokens individuels (ex: "850k;v2" → {"850k","v2"})
                for tok in str(d["names"][j]).split(";"):
                    t = tok.strip()
                    if t:
                        results[site_gi].add(t)

    return results

def _load_sites_reference(path: str) -> pl.DataFrame:
    """Charge un fichier de référence de sites (.bed ou .parquet)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(
            path, separator="\t", comment_prefix="#",
            null_values=["NA", "NaN", "."], has_header=False,
        )
        cols = df.columns
        if len(cols) < 3:
            raise ValueError(
                f"Fichier de sites {path} : au moins 3 colonnes (chr, start, end) attendues, "
                f"{len(cols)} trouvée(s)."
            )
        df = df.rename({cols[0]: "chr", cols[1]: "start", cols[2]: "end"})

    rename_map = {}
    if "chr" not in df.columns and "chrom" in df.columns:
        rename_map["chrom"] = "chr"
    if "start" not in df.columns and "chromStart" in df.columns:
        rename_map["chromStart"] = "start"
    if "end" not in df.columns and "chromEnd" in df.columns:
        rename_map["chromEnd"] = "end"
    if rename_map:
        df = df.rename(rename_map)

    required = ["chr", "start", "end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier de sites {path} : {', '.join(missing)}")

    n_before = df.height
    # NB : on ne normalise PAS le préfixe "chr" ici. Les bedMethyl produits par Modkit portent
    # des noms de chromosome préfixés (chr1, chr2, ...) ; site_id doit rester cohérent entre le
    # référentiel de sites et les parquets patients, sous peine de jointure silencieusement vide
    # (voir la vérification de cohérence ajoutée dans build_cohort_matrix).
    df = df.select(["chr", "start", "end"]).with_columns(
        pl.col("chr").cast(pl.Utf8).str.strip_chars(),
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
    )
    df = df.filter(
        pl.col("start").is_not_null()
        & pl.col("end").is_not_null()
        & pl.col("chr").is_not_null()
        & (pl.col("start") >= 0)
    )
    n_excluded = n_before - df.height
    if n_excluded > 0:
        log.info(
            "Référentiel de sites : %d/%d lignes exclues (coordonnées invalides)",
            n_excluded, n_before,
        )

    if df.height == 0:
        raise ValueError(f"Aucun site valide trouvé dans {path}")

    sites = df.unique().sort(["chr", "start"])
    sites = sites.with_row_index(name="site_code")
    sites = sites.with_columns(
        (pl.col("chr") + ":" + pl.col("start").cast(pl.Utf8)).alias("site_id")
    )
    return sites


# ========================= STEP 1 : parse_sample ==========

def cmd_parse_sample(args):
    """
    Lit un fichier bedMethyl produit par Modkit.
    Garde uniquement 5mC (m).
    Si + et - existent pour le même (chrom, start, mod_code) → calcule la moyenne pondérée par N_valid.
    site_id = "chrom:start"
    Écrit {sample}.parquet (colonnes : site_id, chrom, start, strand, mod_code, N_valid, fraction).
    """
    t0 = time.time()
    # Nom unique du patient = session_barcode
    sample_name = args.sample_name
    log.info("parse_sample [%s] : lecture de %s", sample_name, args.input)
    os.makedirs(args.outdir, exist_ok=True)

    if args.etape == "before_imputation":
        # Colonnes bedMethyl (0-indexed) que l'on utilise
        # col 0  chrom
        # col 1  start
        # col 2  end
        # col 3  mod_code
        # col 5  strand
        # col 9  N_valid
        # col 10 fraction
        # USED_COLS = [0, 1, 2, 3, 5, 9, 10]
        # COL_NAMES = ["chrom", "start", "end", "mod_code", "strand", "N_valid", "fraction"]

        log.info("  lecture du fichier bedMethyl Modkit (avant imputation)")
        try:
            df = pd.read_csv(
                args.input,
                sep='\t',
                header=None,
                comment="#",
                usecols=USED_COLS,
                names=COL_NAMES,
                dtype={
                    "chrom":    str,
                    "start":    np.int64,
                    "end":      np.int64,
                    "mod_code": str,
                    "strand":   str,
                    "N_valid":  np.int64,
                    "fraction": float,
                },
                low_memory=False,
            )
        except Exception as e:
            log.error("Impossible de lire %s : %s", args.input, e)
            sys.exit(1)

    if args.etape == "after_imputation":
        log.info("  lecture du fichier bedMethyl Modkit (après imputation)")
        USED_COLS = [0, 1, 2, 3, 5, 6]
        COL_NAMES = ["chrom", "start", "end", "strand", "fraction", "N_valid"]

        try:
            df = pd.read_csv(
                args.input,
                sep='\t',
                header=1,
                comment="#",
                usecols=USED_COLS,
                names=COL_NAMES,
                dtype={
                    "chrom":    str,
                    "start":    np.int64,
                    "end":      np.int64,
                    "strand":   str,
                    "fraction": float,
                    "N_valid":  np.int64,
                },
                low_memory=False,
            )
        except Exception as e:
            log.error("Impossible de lire %s : %s", args.input, e)
            sys.exit(1)


    log.info("  %d lignes lues", len(df))

    # ── Filtre mod_code : 5mC uniquement ────────────────────
    if args.etape == "before_imputation":
        before = len(df)
        df = df[df["mod_code"] == "m"].copy()
        log.info(
            "  après filtre mod_code (m uniquement) : %d lignes (supprimé %d)",
            len(df), before - len(df),
        )

        if df.empty:
            log.warning(
                "  ATTENTION : aucun site 5mC dans %s."
                " Vérifiez que Modkit a bien produit des codes 'm'.",
                args.input,
            )


    # ── Normalisation fraction → [0,1] ──────────────────────
    # Modkit sort des valeurs entre 0 et 100
    log.info(
            "  normalisation des valeurs de méthylation vers [0,1]"
        )
    df["fraction"] = df["fraction"] / 100.0

    # Pour les sorties GIMMEcpg
    is_prefixed = df["chrom"].str.startswith("chr")
    is_mito = df["chrom"].isin(["MT", "M"])
    
    df["chrom"] = np.select(
        [is_prefixed, is_mito],
        [df["chrom"], "chrM"],
        default="chr" + df["chrom"],
    )


    # Normalisation de la coordonnée sur la base de la convention CpG (Watson start) :
    # pour un CpG à la position p (brin +), la cytosine du brin - est rapportée par
    # Modkit en position p+1. On ramène systématiquement à p pour que (i) les deux
    # brins d'un même CpG partagent le même site_id, permettant leur fusion, et
    # (ii) ce site_id coïncide avec le "start" du BED de référence (dinucléotide
    # CpG, convention 0-based demi-ouverte [p, p+2)).

    if args.etape == "before_imputation":
        if not df["strand"].isin(["+", "-"]).all():
            bad = df.loc[~df["strand"].isin(["+", "-"]), "strand"].unique()
            log.error("  Valeurs de strand inattendues (attendu '+'/'-') : %s", bad)
            sys.exit(1)

        log.info(
            "  normalisation des coordonnées start pour la convention CpG (Watson start)"
        )

        df["start"] = np.where(df["strand"] == "-", df["start"] - 1, df["start"])
    


    # ── Résolution brin +/- : garde la moyenne de méthylation pondérée par N_valid ─────
    # Implémentation vectorisée (groupby().agg()) : évite groupby().apply(lambda -> Série),
    # qui (1) est lente sur un gros bedMethyl et (2) lève un KeyError sous pandas >= 2.2 / 3.0
    # car les colonnes de la clé de groupe sont exclues du sous-DataFrame passé à la lambda
    # quand as_index=False (x["chrom"] n'existe alors plus dans le groupe).

    if args.etape == "before_imputation":
        key = ["chrom", "start"]
    else:
        key = ["chrom", "start", "mod_code"]
    n_before_dedup = len(df)
    df["_weighted_fraction"] = df["fraction"] * df["N_valid"]
    agg = df.groupby(key, as_index=False).agg(
        N_valid=("N_valid", "sum"),
        _weighted_sum=("_weighted_fraction", "sum"),
    )
    agg["fraction"] = agg["_weighted_sum"] / agg["N_valid"]
    df = agg.drop(columns="_weighted_sum")
    # La région CpG est représentée par un intervalle [start, start+2) ;
    # on la recrée explicitement après le regroupement sur chrom/start/mod_code.
    df["end"] = df["start"] + 2
    n_removed_dedup = n_before_dedup - len(df)
    if n_removed_dedup > 0:
        log.info(
            "  regroupement brins +/- : %d lignes (supprimé %d)",
            len(df), n_removed_dedup,
        )

    # ── Construction site_id ─────────────────────────────────
    df["site_id"] = (
        df["chrom"].astype(str) + ":" +
        df["start"].astype(str)
    )

    if args.etape == "after_imputation":
        df["mod_code"] = pd.Series(["m"] * len(df), index=df.index)  # on force le mod_code à 'm' pour la suite, car on ne traite que 5mC pour GIMMEcpg


    # ── Vérification unicité ─────────────────────────────────
    n_dup = df["site_id"].duplicated().sum()
    if n_dup > 0:
        log.error(
            "  %d site_id dupliqués après déduplication — problème inattendu. "
            "Arrêt.",
            n_dup,
        )
        sys.exit(1)

    # ── Résumé ───────────────────────────────────────────────
    log.info(
        "  mod_code='m' : %d sites, N_valid médian=%.0f, beta médian=%.3f",
        len(df), df["N_valid"].median(), df["fraction"].median(),
    )

    # ── Écriture parquet ─────────────────────────────────────
    out_cols = ["site_id", "chrom", "start", "end", "mod_code", "N_valid", "fraction"]
    out = os.path.join(args.outdir, f"{sample_name}.parquet")
    pl.from_pandas(df[out_cols]).write_parquet(out, compression="zstd")
    log.info(
        "parse_sample [%s] : %d sites écrits → %s (%s)",
        sample_name, len(df), out, _fmt_duration(time.time() - t0),
    )


# ========================= STEP 2 : build_cohort_matrix ===

def cmd_build_cohort_matrix(args):
    """
    Fusionne les parquets des patients.
    Les sites sont définis par le fichier de référence passé en entrée ; si un patient n'a pas de valeur
    pour un site, beta=NaN et cov=NaN. Les patients sous le seuil min_cov reçoivent également NaN.

    Seul le mod_code 'm' (5mC) est traité.

    Sortie : cohort_matrix.parquet
    Colonnes : site_id, chrom, start, end, mod_code, beta_<s1>, cov_<s1>, beta_<s2>, cov_<s2>, ...
    Les valeurs beta sont en pourcentage (0-100), arrondi à 2 décimales.
    """
    import math

    t0 = time.time()
    sample_names = args.sample_names  # liste dans le même ordre que --input
    n_patients   = len(args.input)

    log.info(
        "build_cohort_matrix : %d patients ",
        n_patients,
    )
    if n_patients != len(sample_names):
        log.error(
            "--input (%d fichiers) et --sample-names (%d noms) doivent avoir la même longueur.",
            n_patients, len(sample_names),
        )
        sys.exit(1)

    # ── Chargement des sites de la matrice─────────────────────
    sites_df = _load_sites_reference(args.sites_file)
    log.info("Sites de référence chargés : %d sites", sites_df.height)

    # ── Chargement de tous les parquets ──────────────────────
    frames: dict = {}  # sample_name → DataFrame pandas
    for path, sname in zip(args.input, sample_names):
        log.info("  chargement %s …", path)
        df = pl.read_parquet(path).to_pandas()
        expected = {"site_id", "chrom", "start", "end", "mod_code", "N_valid", "fraction"}
        missing = expected - set(df.columns)
        if missing:
            log.error("  Colonnes manquantes dans %s : %s", path, missing)
            sys.exit(1)
        # Seul 5mC est attendu à ce stade, mais on filtre par sécurité
        df = df[df["mod_code"] == "m"].copy()
        frames[sname] = df

    # ── Vérification défensive : cohérence du préfixe chromosomique ─────────
    # Un mismatch "chr1" (patients) vs "1" (sites-file), ou l'inverse, produit une jointure
    # site_id silencieusement vide (matrice entièrement NaN), sans qu'aucune exception ne soit
    # levée. On échoue explicitement ici plutôt que de laisser passer un résultat corrompu.
    def _has_chr_prefix(chrom_values) -> bool:
        sample = next((c for c in chrom_values if c), None)
        return bool(sample) and str(sample).lower().startswith("chr")

    sites_chroms = sites_df.get_column("chr").unique().to_list()
    sites_has_prefix = _has_chr_prefix(sites_chroms)
    for sname, df in frames.items():
        patient_chroms = df["chrom"].unique().tolist()
        patient_has_prefix = _has_chr_prefix(patient_chroms)
        if patient_has_prefix != sites_has_prefix:
            log.error(
                "  Incohérence de préfixe chromosomique entre --sites-file (ex: %s) et le "
                "patient '%s' (ex: %s). La jointure sur site_id serait silencieusement vide. "
                "Harmonisez le préfixe 'chr' entre le référentiel de sites et les bedMethyl.",
                sites_chroms[0] if sites_chroms else "?",
                sname,
                patient_chroms[0] if patient_chroms else "?",
            )
            sys.exit(1)

    # ── Comptage des patients bien couverts par site ─────────
    # On compte combien de patients couvrent chaque site via un Counter.
    from collections import Counter
    coverage_count: Counter = Counter()   # site_id → nb de patients couverts
    coverage_flags: dict    = {}          # sname   → set de site_id bien couverts

    if args.min_cov > 1:
        for sname, df in frames.items():
            covered = set(df.loc[df["N_valid"] >= args.min_cov, "site_id"])
            coverage_flags[sname] = covered
            coverage_count.update(covered)
            log.info(
                "  patient='%s' : %d sites avec N_valid≥%d / %d présents",
            sname, len(covered), args.min_cov, len(df),
            )

    # ── Assemblage de la matrice ─────────────────────────────
    # Les lignes de la matrice sont strictement les sites fournis dans --sites-file.
    # Construire un DataFrame pandas avec les colonnes attendues (site_id, chrom, start, end, mod_code).
    coord_pd = sites_df.select(["chr", "start", "end"]).to_pandas()
    coord_pd = coord_pd.rename(columns={"chr": "chrom"})
    # Assurer mod_code 'm' (5mC) pour la matrice de cohorte.
    coord_pd["mod_code"] = "m"
    # strand inconnu dans le fichier sites -> utiliser '.'
    coord_pd["strand"] = "."
    coord_pd["site_id"] = coord_pd["chrom"].astype(str) + ":" + coord_pd["start"].astype(str)
    merged = coord_pd[["site_id", "chrom", "start", "end", "strand", "mod_code"]].copy()
    merged = merged.reset_index(drop=True)

    for sname, df in frames.items():
        sub = df[["site_id", "N_valid", "fraction"]].copy()
        # Masquer les patients sous-couverts sur ce site : NaN pour beta et cov.
        # Cela concerne les patients avec N_valid < min_cov ET les patients pour lesquels le site est absent (géré par le left join ci-dessous).
        if args.min_cov > 1:
            below_cov_mask = sub["N_valid"] < args.min_cov
            sub.loc[below_cov_mask, "fraction"] = np.nan
            sub.loc[below_cov_mask, "N_valid"]  = np.nan

        sub = sub.rename(columns={
            "fraction": f"beta_{sname}",
            "N_valid":  f"cov_{sname}",
        })
        merged = merged.merge(sub, on="site_id", how="left")
        # Les sites absents du patient → NaN (déjà géré par le left join)

    # ── Conversion beta → pourcentage arrondi à 2 décimales ─
    # On ignore les NaN (sites non couverts).
    for sname in sample_names:
        bcol = f"beta_{sname}"
        if bcol in merged.columns:
            merged[bcol] = (merged[bcol] * 100.0).round(2)

    # ── Rapport des NaN ─────────────────────────────────────
    for sname in sample_names:
        n_nan = merged[f"beta_{sname}"].isna().sum()
        n_total = len(merged)
        if n_nan > 0:
            log.info(
                "  patient='%s' : %d / %d sites avec beta=NaN (couverture insuffisante ou site absent)",
                sname, n_nan, n_total,
            )

    # ── Exclure les sites qui ne sont couverts par AUCUN patient ──────
    # Pour le test de couverture, on regarde les colonnes cov_<sample> : si au moins une
    # valeur de couverture est non NaN, le site est couvert par ce patient.
    cov_cols = [f"cov_{sname}" for sname in sample_names]
    covered_mask = merged[cov_cols].notna().any(axis=1)
    n_total_sites = len(merged)
    n_uncovered = int((~covered_mask).sum())
    if n_uncovered > 0:
        excluded = merged.loc[~covered_mask, ["chrom", "start", "site_id"]].copy()
        # end = start + 2 (CpG dinucleotide)
        excluded["end"] = excluded["start"] + 2
        excluded_bed = os.path.join(args.outdir, "excluded_sites_from_sites_file.bed")
        os.makedirs(args.outdir, exist_ok=True)
        excluded.to_csv(excluded_bed, sep="\t", header=False, index=False, columns=["chrom", "start", "end", "site_id"])
        log.info(
            "  %d sites (%.2f%%) du fichier --sites-file ne sont couverts par aucun patient.\n  -> écrits : %s",
            n_uncovered, 100.0 * n_uncovered / n_total_sites, excluded_bed,
        )
    else:
        log.info("  tous les %d sites du fichier --sites-file sont couverts par au moins un patient", n_total_sites)

    # Conserver uniquement les sites couverts par au moins un patient
    merged = merged.loc[covered_mask].reset_index(drop=True)

    log.info(
        "  matrice finale : %d sites × %d patients (5mC uniquement)",
        len(merged), len(sample_names),
    )

    # ── Ordre des colonnes ───────────────────────────────────
    coord_cols = ["site_id", "chrom", "start", "end", "mod_code"]
    patient_cols = []
    for sname in sample_names:
        patient_cols += [f"beta_{sname}", f"cov_{sname}"]
    merged = merged[coord_cols + patient_cols]

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "cohort_matrix.parquet")
    pl.from_pandas(merged).write_parquet(out, compression="zstd")
    log.info(
        "build_cohort_matrix → %s (%s)",
        out, _fmt_duration(time.time() - t0),
    )


# ========================= STEP 3 : assign_origins ========

def cmd_assign_origins(args):
    """
    Assigne l'origine BED source à chaque site.
    Un site peut chevaucher plusieurs régions → origines concaténées triées.
    Sites sans aucun hit : origine = "unknown" (conservés, signalés).
    """
    t0 = time.time()
    log.info("assign_origins : chargement de cohort_matrix.parquet…")

    df = pl.read_parquet(args.cohort_matrix).to_pandas()
    log.info("  %d sites chargés", len(df))

    # end = start + 2 (dinucléotide CpG)
    df["end"] = df["start"] + 2

    # ── Index BED source ─────────────────────────────────────
    log.info("assign_origins : chargement du BED source %s…", args.source_bed)
    source_raw = pd.read_csv(
        args.source_bed, sep="\t", header=None, comment="#", encoding="latin-1",
        names=["chrom", "start", "end", "origin"],
        dtype={"chrom": str, "origin": str},
    )
    source_raw["start"] = pd.to_numeric(source_raw["start"], errors="coerce").fillna(0).astype(np.int64)
    source_raw["end"]   = pd.to_numeric(source_raw["end"],   errors="coerce").fillna(0).astype(np.int64)

    source_idx: dict = {}
    for ch, grp in source_raw.groupby("chrom", sort=False):
        g = grp.sort_values("start")
        source_idx[str(ch)] = {
            "starts": g["start"].to_numpy(dtype=np.int64),
            "ends":   g["end"].to_numpy(dtype=np.int64),
            "names":  g["origin"].to_numpy(dtype=object),
        }

    # ── Calcul de toutes les origines pour chaque site ───────
    all_origin_sets = _vec_all_overlaps(df, source_idx)

    origins = []
    n_unknown = 0
    for s in all_origin_sets:
        if not s:
            origins.append("unknown")
            n_unknown += 1
        else:
            origins.append(";".join(sorted(s)))

    df["origin"] = origins
    log.info(
        "assign_origins : %d sites sans hit source (unknown) sur %d total (%.1f %%)",
        n_unknown, len(df), 100 * n_unknown / len(df) if len(df) else 0,
    )
    log.info(
        "assign_origins : répartition origine (top 15)\n%s",
        df["origin"].value_counts().head(15).to_string(),
    )

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "cohort_with_origins.parquet")
    # Supprimer la colonne end temporaire avant d'écrire
    df = df.drop(columns=["end"])
    pl.from_pandas(df).write_parquet(out, compression="zstd")
    log.info("assign_origins → %s (%s)", out, _fmt_duration(time.time() - t0))


# ========================= STEP 4 : annotate_genomic ======

def cmd_annotate_genomic(args):
    """
    Annote chaque site par priorité :
    Promoter/TSS > Exon > Intron > cCRE > Intergenic

    Colonnes ajoutées : feature_label, gene_name, distance_TSS,
                        ccre_accession, ccre_type
    """
    t0 = time.time()
    log.info("annotate_genomic : chargement…")

    df = pl.read_parquet(args.cohort_with_origins).to_pandas()
    df["end"] = df["start"] + 2
    log.info("  %d sites", len(df))

    ref_tss     = _build_interval_index(args.tss_bed,     name_col=3)
    ref_exons   = _build_interval_index(args.exons_bed,   name_col=3)
    ref_introns = _build_interval_index(args.introns_bed, name_col=3)
    ref_ccre    = _build_interval_index(args.ccre_bed,    name_col=3)

    # Distance TSS (col 5 optionnelle)
    ref_tss_dist = {}
    try:
        ref_tss_dist = _build_interval_index(args.tss_bed, name_col=5)
    except Exception:
        log.info("  colonne distance TSS (col 5) absente, ignorée")

    # ccre_type (col 9 optionnelle)
    ref_ccre_type = {}
    try:
        ref_ccre_type = _build_interval_index(args.ccre_bed, name_col=9)
    except Exception:
        log.info("  colonne ccre_type (col 9) absente, ignorée")

    n = len(df)
    feature_label  = np.full(n, "Intergenic", dtype=object)
    gene_name      = np.full(n, "",           dtype=object)
    distance_TSS   = np.full(n, "",           dtype=object)
    ccre_accession = np.full(n, "",           dtype=object)
    ccre_type_arr  = np.full(n, "",           dtype=object)
    annotated      = np.zeros(n, dtype=bool)

    for label, ref in [
        ("Promoter/TSS", ref_tss),
        ("Exon",         ref_exons),
        ("Intron",       ref_introns),
        ("cCRE",         ref_ccre),
    ]:
        log.info("  overlap %s…", label)
        mask, names = _vec_first_overlap(df, ref)
        new_hits = mask & ~annotated
        feature_label[new_hits] = label

        if label in ("Promoter/TSS", "Exon", "Intron"):
            gene_name[new_hits] = names[new_hits]
            if label == "Promoter/TSS" and ref_tss_dist:
                _, dist_names = _vec_first_overlap(df, ref_tss_dist)
                distance_TSS[new_hits] = dist_names[new_hits]
        elif label == "cCRE":
            ccre_accession[new_hits] = names[new_hits]
            if ref_ccre_type:
                _, ctype = _vec_first_overlap(df, ref_ccre_type)
                ccre_type_arr[new_hits] = ctype[new_hits]

        annotated |= new_hits
        log.info(
            "    %d nouveaux hits → total annoté %d / %d",
            new_hits.sum(), annotated.sum(), n,
        )

    df["feature_label"]  = feature_label
    df["gene_name"]      = gene_name
    df["distance_TSS"]   = distance_TSS
    df["ccre_accession"] = ccre_accession
    df["ccre_type"]      = ccre_type_arr
    df = df.drop(columns=["end"])

    log.info(
        "annotate_genomic : répartition\n%s",
        df["feature_label"].value_counts().to_string(),
    )

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "cohort_genomic.parquet")
    pl.from_pandas(df).write_parquet(out, compression="zstd")
    log.info("annotate_genomic → %s (%s)", out, _fmt_duration(time.time() - t0))


# ========================= STEP 5 : export_tsv ============

def cmd_export_tsv(args):
    """
    Exporte cohort_genomic.parquet → methylation_cohort.tsv
    Format : une ligne par (site_id, mod_code)
    Colonnes : chrom, start, strand, mod_code, <annotations...>,
               beta_<s1>, cov_<s1>, beta_<s2>, cov_<s2>, ...
    Les valeurs NaN (couverture insuffisante) sont exportées comme chaîne vide.
    """
    t0 = time.time()
    log.info("export_tsv : chargement de %s…", args.cohort_genomic)

    df = pl.read_parquet(args.cohort_genomic).to_pandas()
    log.info("  %d lignes, %d colonnes", *df.shape)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "methylation_cohort.tsv")
    # na_rep="NaN" : les valeurs manquantes (couverture insuffisante) apparaissent explicitement comme "NaN" dans le TSV, ce qui permet des tests directs
    # en awk/grep (ex: awk '$7 == "NaN"') et évite toute ambiguïté avec une cellule vide.
    df.to_csv(out, sep="\t", index=False, na_rep="NaN")
    log.info(
        "export_tsv → %s (%d lignes, %s)",
        out, len(df), _fmt_duration(time.time() - t0),
    )


# ========================= ARGPARSE ======================

def parse_args():
    p = argparse.ArgumentParser(
        description="Methylation cohort pipeline (bedMethyl → matrice filtrée)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── 1. parse_sample ──────────────────────────────────────
    s1 = sub.add_parser(
        "parse_sample",
        help="Lit un bedMethyl Modkit → {sample}.parquet",
    )
    s1.add_argument("--input",       "-i", required=True,
                    help="Fichier bedMethyl d'un patient")
    s1.add_argument("--sample-name", required=True,
                    help="Identifiant du barcode (ex: barcode05). "
                         "Le nom final du patient sera session_barcode.")
    s1.add_argument("--outdir",      "-o", required=True,
                    help="Dossier de sortie")
    s1.add_argument("--etape", choices=["before_imputation", "after_imputation"], default="before_imputation",
                    help="Étape du pipeline (défaut : before_imputation).")

    # ── 2. build_cohort_matrix ───────────────────────────────
    s2 = sub.add_parser(
        "build_cohort_matrix",
        help="Fusionne les parquets patients et applique le filtre de couverture semi-strict",
    )
    s2.add_argument("--input", "-i", nargs="+", required=True,
                    help="Liste des {sample}.parquet (même ordre que --sample-names)")
    s2.add_argument("--sample-names", nargs="+", required=True,
                    help="Identifiants des patients (même ordre que --input)")
    s2.add_argument("--sites-file", required=True,
                   help="Fichier de référence des sites de la matrice (.bed ou .parquet)")
    s2.add_argument("--min-cov", type=int, default=1,
                    help="Coverage minimal par patient (défaut : 1). "
                         "Un patient est considéré 'couvert' sur un site si "
                         "N_valid ≥ min_cov. Les patients sous le seuil reçoivent "
                         "beta=NaN, cov=NaN.")
    s2.add_argument("--type-output", choices=["parquet", "tsv"], default="parquet",
                    help="Format de sortie (défaut : parquet).")
    s2.add_argument("--outdir", "-o", required=True,
                    help="Dossier de sortie → cohort_matrix.parquet")

    # ── 3. assign_origins ────────────────────────────────────
    s3 = sub.add_parser(
        "assign_origins",
        help="Assigne l'origine BED source à chaque site",
    )
    s3.add_argument("--cohort-matrix", required=True,
                    help="cohort_matrix.parquet")
    s3.add_argument("--source-bed",    required=True,
                    help="BED source (chrom start end origine)")
    s3.add_argument("--outdir", "-o",  required=True,
                    help="Dossier de sortie → cohort_with_origins.parquet")

    # ── 4. annotate_genomic ──────────────────────────────────
    s4 = sub.add_parser(
        "annotate_genomic",
        help="Annote les sites avec les régions génomiques",
    )
    s4.add_argument("--cohort-with-origins", required=True,
                    help="cohort_with_origins.parquet")
    s4.add_argument("--tss-bed",     required=True,
                    help="BED TSS (chrom start end gene_name [strand distance_TSS])")
    s4.add_argument("--exons-bed",   required=True,
                    help="BED exons (chrom start end gene_name)")
    s4.add_argument("--introns-bed", required=True,
                    help="BED introns (chrom start end gene_name)")
    s4.add_argument("--ccre-bed",    required=True,
                    help="BED cCRE (chrom start end accession ... [ccre_type col9])")
    s4.add_argument("--outdir", "-o", required=True,
                    help="Dossier de sortie → cohort_genomic.parquet")

    # ── 5. export_tsv ────────────────────────────────────────
    s5 = sub.add_parser(
        "export_tsv",
        help="Exporte cohort_genomic.parquet → methylation_cohort.tsv",
    )
    s5.add_argument("--cohort-genomic", required=True,
                    help="cohort_genomic.parquet")
    s5.add_argument("--outdir", "-o",   required=True,
                    help="Dossier de sortie → methylation_cohort.tsv")

    return p.parse_args()


# ========================= MAIN ==========================

def main():
    args = parse_args()
    dispatch = {
        "parse_sample":        cmd_parse_sample,
        "build_cohort_matrix": cmd_build_cohort_matrix,
        "assign_origins":      cmd_assign_origins,
        "annotate_genomic":    cmd_annotate_genomic,
        "export_tsv":          cmd_export_tsv,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()