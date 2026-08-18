#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distribution_meth.py — Diagnostic de la distribution de méthylation
===========================================================================

Prend en entrée methylation_cohort.tsv (une ligne par site CpG, colonnes
beta_<sample> et cov_<sample>) et produit trois figures de diagnostic :

  1. distribution_beta_full.png
        Histogramme complet des valeurs beta [0, 1].
        Les pics exacts à 0 et 1 sont représentés comme des barres
        distinctes (sites non méthylés / méthylés à 100%) avec leur
        pourcentage annoté. Le reste [0, 1[ est subdivisé en bins réguliers.

  2. distribution_beta_zoom.png
        Même histogramme, mais les valeurs beta = 0 et beta = 1 sont
        exclues pour visualiser la structure de la distribution intermédiaire
        sans écrasement d'échelle. Les pourcentages exclus sont rappelés
        dans un encadré sur la figure.

  3. distribution_coverage.png
        Distribution de la profondeur (nombre de reads) par site et par
        patient, en échelle log sur l'axe X (la profondeur ONT suit
        typiquement une distribution log-normale ou en loi de puissance
        avec une queue droite étendue). Statistiques clés annotées :
        médiane, percentiles 10/90.

  4. distribution_variance.png
        Distribution de la variance des valeurs beta par site, calculée sur
        les patients disponibles pour chaque site. Cette figure permet de
        repérer les sites très variables versus quasi constants dans la
        cohorte.

Implémentation :
    Les colonnes beta_* et cov_* sont traitées une par une (boucle sur
    les patients) et les comptes de bins sont accumulés incrémentalement
    en numpy — la matrice complète n'est jamais chargée entièrement en
    mémoire. Compatible avec des cohortes de plusieurs millions de sites
    CpG × dizaines de patients.
"""

import os
import sys
import logging
import argparse

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ========================= ARGPARSE ==========================

def parse_args():
    p = argparse.ArgumentParser(
        description="Distribution de méthylation et de profondeur — 3 figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cohort-tsv", required=True,
                   help="methylation_cohort.tsv (colonnes beta_<sample> / cov_<sample>)")
    p.add_argument("--min-cov", type=int, default=5,
                   help="Seuil de profondeur minimale pour la figure de "
                        "profondeur (ligne verticale annotée, défaut : 5)")
    p.add_argument("--n-beta-bins", type=int, default=100,
                   help="Nombre de bins pour la zone (0, 1) dans les "
                        "histogrammes beta (défaut : 100)")
    p.add_argument("--n-variance-bins", type=int, default=50,
                   help="Nombre de bins pour la distribution de variance "
                        "par site (défaut : 50)")
    p.add_argument("--tag", default=None,
                   help="Suffixe ajouté aux noms de fichiers de sortie")
    p.add_argument("--outdir", "-o", required=True)
    return p.parse_args()


# ========================= HELPERS ===========================

def _suffix(tag):
    return f"_{tag}" if tag else ""


def _detect_columns(columns):
    """
    Détecte les colonnes beta_<sample> et cov_<sample> appariées.
    Retourne (beta_cols, cov_cols, samples).
    """
    samples = []
    for c in columns:
        if c.startswith("beta_"):
            s = c[len("beta_"):]
            if f"cov_{s}" in columns:
                samples.append(s)
    if not samples:
        log.error(
            "Aucune colonne beta_<sample>/cov_<sample> détectée. "
            "Vérifiez le format du TSV."
        )
        sys.exit(1)
    beta_cols = [f"beta_{s}" for s in samples]
    cov_cols  = [f"cov_{s}"  for s in samples]
    return beta_cols, cov_cols, samples


def _estimate_beta_median(n_zero, n_one, hist_inter, edges_inter):
    """Estime la médiane des valeurs beta à partir de l'histogramme."""
    n_total = n_zero + n_one + int(hist_inter.sum())
    if n_total == 0:
        return np.nan

    values = np.concatenate(([0.0], (edges_inter[:-1] + edges_inter[1:]) / 2, [100.0]))
    counts = np.concatenate(([n_zero], hist_inter.astype(np.int64), [n_one]))
    cumulative = np.cumsum(counts)
    target = n_total / 2.0

    idx = np.searchsorted(cumulative, target, side="left")
    if idx == 0:
        return 0.0
    if idx >= len(values):
        return 100.0

    prev_cum = cumulative[idx - 1] if idx > 0 else 0.0
    prev_val = values[idx - 1] if idx > 0 else 0.0
    curr_cum = cumulative[idx]
    curr_val = values[idx]

    if curr_cum <= prev_cum:
        return curr_val

    fraction = (target - prev_cum) / (curr_cum - prev_cum)
    return prev_val + fraction * (curr_val - prev_val)


def _accumulate_beta(df: pl.DataFrame, beta_cols: list, n_bins: int):
    """
    Calcule incrémentalement les comptes nécessaires aux histogrammes beta.

    Retourne :
        n_zero   : nombre total de valeurs beta == 0.0 (exactement)
        n_one    : nombre total de valeurs beta == 100.0 (exactement)
        n_inter  : nombre total de valeurs 0 < beta < 100
        n_nan    : nombre total de NaN
        hist_inter : tableau de comptes (n_bins,) pour la zone (0, 100)
        edges_inter: bords des bins de hist_inter
        mean_beta : moyenne des valeurs beta valides
        median_beta : médiane estimée des valeurs beta valides

    Traitement colonne par colonne pour limiter la mémoire.
    """
    edges_inter = np.linspace(0.0, 100.0, n_bins + 1)
    hist_inter  = np.zeros(n_bins, dtype=np.int64)
    n_zero = n_one = n_inter = n_nan = 0
    sum_beta = 0.0
    n_valid = 0

    for col in beta_cols:
        vals = df[col].cast(pl.Float64, strict=False).to_numpy()

        mask_nan  = np.isnan(vals)
        mask_zero = (vals == 0.0) & ~mask_nan
        mask_one  = (vals == 100.0) & ~mask_nan
        mask_int  = ~mask_nan & ~mask_zero & ~mask_one

        n_nan  += int(mask_nan.sum())
        n_zero += int(mask_zero.sum())
        n_one  += int(mask_one.sum())
        n_inter += int(mask_int.sum())

        if mask_int.any():
            c, _ = np.histogram(vals[mask_int], bins=edges_inter)
            hist_inter += c

        valid_vals = vals[~mask_nan]
        if valid_vals.size:
            sum_beta += float(valid_vals.sum())
            n_valid += int(valid_vals.size)

    mean_beta = sum_beta / n_valid if n_valid else np.nan
    median_beta = _estimate_beta_median(n_zero, n_one, hist_inter, edges_inter)

    return n_zero, n_one, n_inter, n_nan, hist_inter, edges_inter, mean_beta, median_beta


def _accumulate_coverage(df: pl.DataFrame, cov_cols: list):
    """
    Collecte toutes les valeurs de profondeur (entières, non-NaN)
    dans un tableau numpy plat pour le diagnostic de distribution.

    La couveprofondeurrture est stockée en entier dans le TSV (nombre de reads).
    On utilise un histogramme log-espacé pour couvrir la gamme typique
    ONT [1, ~500x] sans saturation d'échelle.

    Retourne (all_cov, n_zero_cov) où :
        all_cov     : array 1D des profondeurs > 0
        n_zero_cov  : nombre de positions à profondeur nulle ou NaN
    """
    all_cov    = []
    n_zero_cov = 0

    for col in cov_cols:
        vals = df[col].cast(pl.Float64, strict=False).to_numpy()
        mask_valid = ~np.isnan(vals) & (vals > 0)
        n_zero_cov += int((~mask_valid).sum())
        if mask_valid.any():
            all_cov.append(vals[mask_valid].astype(np.int64))

    if all_cov:
        return np.concatenate(all_cov), n_zero_cov
    return np.array([], dtype=np.int64), n_zero_cov


def _compute_site_variance(df: pl.DataFrame, beta_cols: list):
    """
    Calcule la variance de chaque site (ligne) sur les valeurs beta
    disponibles parmi les patients.

    Les sites entièrement manquants ou ne possédant qu'une seule valeur
    valide n'ont pas de variance calculable ; ils sont exclus du plot.
    """
    site_vars = []

    for row in df.select(beta_cols).iter_rows(named=False):
        vals = np.asarray(row, dtype=np.float64)
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            continue
        site_vars.append(float(np.var(vals[mask], ddof=1)))

    if site_vars:
        return np.asarray(site_vars, dtype=np.float64)
    return np.array([], dtype=np.float64)


# ========================= FIGURES ===========================

def _fig_beta_full(n_zero, n_one, n_inter, n_nan,
                   hist_inter, edges_inter, mean_beta, median_beta,
                   outdir, tag):
    """
    Figure 1 : distribution complète des beta values.

    Structure de l'histogramme :
      - Barre "0" à gauche  : beta == 0 exactement (sites non méthylés)
      - n_bins barres centrales : 0 < beta < 100 (méthylation partielle)
      - Barre "1" à droite  : beta == 1 exactement (sites méthylés à 100%)

    Les NaN sont comptabilisés séparément et rappelés en légende mais
    exclus des pourcentages (ils ne sont pas des mesures de méthylation,
    mais des absences de profondeur).

    Les pourcentages sont calculés sur le total des valeurs non-NaN :
        % = n_catégorie / (n_zero + n_inter + n_one) × 100
    """
    n_total   = n_zero + n_inter + n_one
    if n_total == 0:
        log.error("Aucune valeur beta valide. Arrêt.")
        sys.exit(1)

    pct_zero  = 100 * n_zero  / n_total
    pct_one   = 100 * n_one   / n_total
    pct_inter = 100 * n_inter / n_total

    # Largeur uniforme des bins intermédiaires
    bin_w = edges_inter[1] - edges_inter[0]

    # Barres spéciales 0 et 1 : même largeur que les bins intermédiaires
    # mais positionnées à l'extérieur avec un espace visuel

    fig, ax = plt.subplots(figsize=(10, 5))

    # Bins intermédiaires
    bin_centers = (edges_inter[:-1] + edges_inter[1:]) / 2
    hist_inter_plot = np.where(hist_inter > 0, hist_inter, 1e-3)
    ax.bar(bin_centers, hist_inter_plot, width=bin_w * 0.95,
           color="#3B8BD4", alpha=0.85, label=f"0 < β < 100  ({pct_inter:.1f}%)")

    # Barre beta = 0
    bar0 = ax.bar(0, n_zero, width=bin_w * 0.95,
                  color="#E8593C", alpha=0.9, label=f"β = 0  ({pct_zero:.1f}%)")

    # Barre beta = 100
    bar1 = ax.bar(100, n_one, width=bin_w * 0.95,
                  color="#1D9E75", alpha=0.9, label=f"β = 100  ({pct_one:.1f}%)")

    # Annotation des pourcentages sur les barres extrêmes
    for bar, pct, n in [(bar0, pct_zero, n_zero), (bar1, pct_one, n_one)]:
        h = bar[0].get_height()
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, h * 1.01,
                f"{pct:.1f}%\n(n={n:,})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Axe X : ticks naturels [0, 25, 50, 75, 100] + étiquettes spéciales
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"],
                       fontsize=9)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax.set_xlabel("Valeur beta (pourcentage de reads méthylés)", fontsize=13)
    ax.set_ylabel("Nombre de mesures (sites × patients)", fontsize=13)
    ax.set_title("Distribution des valeurs de méthylation — cohorte complète",
                 fontsize=15, fontweight="bold")

    if not np.isnan(mean_beta):
        ax.axvline(mean_beta, color="#E8593C", linewidth=1.5,
                   linestyle="-", label=f"Moyenne : {mean_beta:.1f}")
    if not np.isnan(median_beta):
        ax.axvline(median_beta, color="#1D9E75", linewidth=1.5,
                   linestyle="--", label=f"Médiane : {median_beta:.1f}")

    # Légende + rappel NaN
    legend = ax.legend(fontsize=11, framealpha=0.8)
    # ax.text(0.99, 0.97,
    #         f"NaN : {n_nan:,}",
    #         transform=ax.transAxes, ha="right", va="top",
    #         fontsize=8, color="grey", style="italic")

    fig.tight_layout()
    out = os.path.join(outdir, f"distribution_beta_full{_suffix(tag)}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def _fig_beta_zoom(n_zero, n_one, n_inter, hist_inter, edges_inter,
                   mean_beta, median_beta, outdir, tag):
    """
    Figure 2 : zoom sur la distribution intermédiaire (0 < beta < 100).

    Les valeurs beta = 0 et beta = 1 sont exclues pour voir la structure
    fine de la méthylation partielle sans écrasement d'échelle par les
    pics extrêmes. Les fractions exclues sont rappelées dans un encadré.

    Cette figure est particulièrement informative pour décider si une
    binarisation à 0.5 est justifiée :
        - Distribution bimodale avec creux marqué à 0.5 → binarisation
          défendable (les valeurs intermédiaires sont rares)
        - Distribution unimodale ou aplatie → binarisation destructrice
          (une fraction significative des sites a une méthylation partielle
          biologiquement réelle, ex. sites en CGI shores, PMDs, DMRs)
    """
    n_total_obs = n_zero + n_one + n_inter
    pct_zero    = 100 * n_zero / n_total_obs if n_total_obs else 0
    pct_one     = 100 * n_one  / n_total_obs if n_total_obs else 0

    bin_centers = (edges_inter[:-1] + edges_inter[1:]) / 2
    bin_w       = edges_inter[1] - edges_inter[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    hist_inter_plot = np.where(hist_inter > 0, hist_inter, 1e-3)
    ax.bar(bin_centers, hist_inter_plot, width=bin_w * 0.95,
           color="#3B8BD4", alpha=0.85)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax.set_xlabel("Valeur beta (pourcentage de reads méthylés)", fontsize=11)
    ax.set_ylabel("Nombre de mesures (site × patient)", fontsize=11)
    ax.set_title(
        "Distribution des valeurs de méthylation — zoom 0 < β < 100\n"
        "(pics β=0 et β=100 exclus)", fontsize=15, fontweight="bold",
    )

    if not np.isnan(mean_beta):
        ax.axvline(mean_beta, color="#E8593C", linewidth=1.5,
                   linestyle="-", label=f"Moyenne : {mean_beta:.1f}")
    if not np.isnan(median_beta):
        ax.axvline(median_beta, color="#1D9E75", linewidth=1.5,
                   linestyle="--", label=f"Médiane : {median_beta:.1f}")

    # Encadré rappelant les fractions exclues
    txt = (f"Exclus de cette figure :\n"
           f"  β = 0 : {pct_zero:.1f}%  (n={n_zero:,})\n"
           f"  β = 100 : {pct_one:.1f}%  (n={n_one:,})")
    ax.text(0.98, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="grey", alpha=0.8))

    ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    out = os.path.join(outdir, f"distribution_beta_zoom{_suffix(tag)}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def _fig_variance(site_vars, n_bins, outdir, tag):
    """
    Figure 4 : distribution de la variance des beta values par site.

    Cette figure met en évidence les sites très variables (potentiellement
    informatifs pour le clustering) et les sites quasi constants qui
    apportent peu d'information discriminante.
    """
    if len(site_vars) == 0:
        log.warning("Aucune variance de site calculable. Figure de variance ignorée.")
        return

    positive_vars = site_vars[site_vars > 0]
    if len(positive_vars) == 0:
        log.warning("Aucune variance strictement positive. Figure de variance ignorée.")
        return

    min_val = float(np.min(positive_vars))
    max_val = float(np.max(positive_vars))
    if max_val <= min_val:
        max_val = min_val * 10 if min_val > 0 else 1.0

    edges = np.logspace(np.log10(max(min_val, 1e-6)), np.log10(max_val), min(n_bins, 100) + 1)
    hist, _ = np.histogram(positive_vars, bins=edges)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.sqrt(edges[:-1] * edges[1:]), hist,
           width=np.diff(edges), color="#F39C12", alpha=0.8, align="center")

    median_var = float(np.median(positive_vars))
    mean_var = float(np.mean(positive_vars))
    p10 = float(np.percentile(positive_vars, 10))
    p90 = float(np.percentile(positive_vars, 90))

    ax.axvline(median_var, color="#E8593C", linewidth=1.5,
               linestyle="-", label=f"Médiane : {median_var:.3f}")
    ax.axvline(mean_var, color="#1D9E75", linewidth=1.5,
               linestyle="--", label=f"Moyenne : {mean_var:.3f}")
    ax.axvspan(p10, p90, alpha=0.12, color="#F39C12",
               label=f"P10–P90 : [{p10:.3f}, {p90:.3f}]")

    ax.set_xscale("log")
    ax.set_xlabel("Variance par site (échelle log)", fontsize=13)
    ax.set_ylabel("Nombre de sites", fontsize=13)
    ax.set_title("Distribution de la variance des beta values par site",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    out = os.path.join(outdir, f"distribution_variance{_suffix(tag)}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def _fig_coverage(all_cov, n_zero_cov, min_cov, outdir, tag):
    """
    Figure 3 : distribution de la profondeur (reads par site par patient).

    Axe X en échelle logarithmique : la profondeur ONT suit typiquement
    une distribution log-normale (log-normale du nombre de reads mappés
    par position selon la profondeur locale). Une échelle linéaire écraserait
    la zone 1-10x et rendrait la queue droite illisible.

    Statistiques annotées :
        - Médiane et moyenne (différentes si distribution asymétrique)
        - Percentiles 10 et 90 (intervalle de profondeur "typique")
        - Fraction de mesures sous min_cov (seuil du filtre semi-strict)
    """
    if len(all_cov) == 0:
        log.warning("Aucune valeur de profondeur valide. Figure de profondeur ignorée.")
        return

    # Bins log-espacés : [1, max_cov] subdivisé en 80 intervalles
    cov_max  = int(all_cov.max())
    cov_min  = max(1, int(all_cov.min()))
    edges    = np.logspace(np.log10(cov_min), np.log10(cov_max + 1), 81)
    hist, _  = np.histogram(all_cov, bins=edges)

    median_cov = float(np.median(all_cov))
    mean_cov   = float(all_cov.mean())
    p10        = float(np.percentile(all_cov, 10))
    p90        = float(np.percentile(all_cov, 90))
    n_total    = len(all_cov) + n_zero_cov
    frac_below = 100 * (all_cov < min_cov).sum() / len(all_cov)

    fig, ax = plt.subplots(figsize=(10, 5))
    bin_centers = np.sqrt(edges[:-1] * edges[1:])   # centre géométrique
    ax.bar(bin_centers, hist, width=np.diff(edges),
           color="#9B59B6", alpha=0.8, align="center")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x)}×"
    ))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))

    # Lignes verticales
    ax.axvline(median_cov, color="#E8593C", linewidth=1.5,
               linestyle="-", label=f"Médiane : {median_cov:.0f}×")
    ax.axvline(mean_cov, color="#EF9F27", linewidth=1.5,
               linestyle="--", label=f"Moyenne : {mean_cov:.0f}×")
    # ax.axvline(min_cov, color="black", linewidth=1.2,
    #            linestyle=":", label=f"Seuil min ({min_cov}×) — {frac_below:.1f}% en dessous")
    # ax.axvspan(cov_min, min_cov, alpha=0.07, color="black")

    # Bande P10-P90
    ax.axvspan(p10, p90, alpha=0.12, color="#9B59B6",
               label=f"P10–P90 : [{p10:.0f}×, {p90:.0f}×]")

    ax.set_xlabel("Profondeur (nombre de reads, échelle log)", fontsize=13)
    ax.set_ylabel("Nombre de mesures (site × patient)", fontsize=13)
    ax.set_title("Distribution de la profondeur — cohorte complète",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8)

    # Encadré statistiques
    txt = (f"n mesures valides : {len(all_cov):,}\n"
           f"n NaN / cov=0 : {n_zero_cov:,}\n"
           f"Max profondeur : {cov_max}×")
    ax.text(0.98, 0.97, txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.8))

    fig.tight_layout()
    out = os.path.join(outdir, f"distribution_coverage{_suffix(tag)}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


# ========================= MAIN ==============================

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    log.info("Chargement de %s…", args.cohort_tsv)
    df = pl.read_csv(
        args.cohort_tsv,
        separator="\t",
        null_values=["NaN", "nan", ""],
        infer_schema_length=1000,
    )
    log.info("  %d lignes × %d colonnes", df.height, df.width)

    # ── Détection des colonnes ───────────────────────────
    beta_cols, cov_cols, samples = _detect_columns(df.columns)
    log.info("  %d patients détectés", len(samples))

    # ── Accumulation des stats beta ──────────────────────
    log.info("  Calcul de la distribution beta (colonne par colonne)…")
    n_zero, n_one, n_inter, n_nan, hist_inter, edges_inter, mean_beta, median_beta = _accumulate_beta(
        df, beta_cols, args.n_beta_bins,
    )
    n_total = n_zero + n_inter + n_one
    log.info(
        "  Beta=0 : %d (%.1f%%) | beta=1 : %d (%.1f%%) | "
        "0<beta<1 : %d (%.1f%%) | NaN : %d",
        n_zero,  100 * n_zero  / n_total if n_total else 0,
        n_one,   100 * n_one   / n_total if n_total else 0,
        n_inter, 100 * n_inter / n_total if n_total else 0,
        n_nan,
    )

    # ── Accumulation des stats profondeur ────────────────
    log.info("  calcul de la distribution de profondeur")
    all_cov, n_zero_cov = _accumulate_coverage(df, cov_cols)

    # ── Calcul de la variance par site ───────────────────
    log.info("  calcul de la distribution de variance par site")
    site_vars = _compute_site_variance(df, beta_cols)

    # ── Production des figures ───────────────────────────
    log.info("  Génération des figures…")
    _fig_beta_full(n_zero, n_one, n_inter, n_nan,
                   hist_inter, edges_inter, mean_beta, median_beta,
                   args.outdir, args.tag)
    _fig_beta_zoom(n_zero, n_one, n_inter,
                   hist_inter, edges_inter, mean_beta, median_beta,
                   args.outdir, args.tag)
    _fig_coverage(all_cov, n_zero_cov, args.min_cov, args.outdir, args.tag)
    _fig_variance(site_vars, args.n_variance_bins, args.outdir, args.tag)

    log.info("Terminé — figures dans %s", args.outdir)


if __name__ == "__main__":
    main()