#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCoA (Principal Coordinates Analysis / Gower) sur une matrice de distance
available-case précalculée, avec :

  1. Analyse de gradient continu (--risk-file) : régression multi-axes PCoA
     ~ variable continue/ordinale, validée par LOOCV (Q2) et testée contre
     le hasard par permutation (sélection de k refaite à chaque permutation).
     Adaptée à une variable comme le score de risque de Manchester.

  2. PERMANOVA (--permanova-column) : test de différence de position entre
     groupes CATÉGORIELS directement sur la matrice de distance, avec p-value
     obtenue par permutation des labels de groupe. Adaptée à une variable
     nominale comme le run de séquençage (effet batch) ou un groupe clinique.


Entrée :
  - --distance-matrix : distance_matrix.tsv produit par
    build_pairwise_clustering.py (matrice patients x patients)
  - --risk-file (optionnel) : TSV à 2 colonnes "patient" et "risk"
    (score continu ou ordinal). Utilisé UNIQUEMENT pour l'analyse de
    gradient.
  - --metadata (optionnel) : TSV avec colonne "patient" (ou index) et une
    ou plusieurs colonnes de métadonnées (age, run, clinical_group, ...).
    Utilisé pour --color-by (coloration des figures) et --permanova-column
    (test PERMANOVA).
  - --color-by (optionnel) : nom de colonne dans --metadata utilisé pour
    colorer le nuage de points PCoA. Si omis et --risk-file fourni, colore
    par le risque.
  - --permanova-column (optionnel) : nom de colonne CATÉGORIELLE dans
    --metadata à tester par PERMANOVA contre la matrice de distance.

Sorties (TSV) :
  - pcoa_eigenvalues.tsv       : variance expliquée par axe + diagnostic
  - pcoa_coordinates.tsv       : coordonnées des patients dans l'espace PCoA
  - axis_risk_correlation.tsv  : corrélation Spearman risque~axe, BH-ajustée
  - risk_gradient_loocv.tsv    : Q2 (LOOCV) pour k=1..max_axes
  - risk_gradient_scores.tsv   : position de chaque patient sur le gradient
  - risk_gradient_direction.tsv: poids de chaque axe dans la direction du gradient
  - permutation_test.tsv       : Q2 observé vs distribution nulle (p-value)
  - permanova_<column>.tsv     : résultats PERMANOVA (pseudo-F, p-value, SS)

Sorties (figures, PNG) :
  - pcoa_scree.png                    : variance expliquée / cumulée par axe
  - pcoa_ordination.png               : nuage de points PCo1 x PCo2 (et PCo1 x
                                         PCo3 si >=3 axes positifs), colorié
                                         selon --color-by ou --risk-file
  - risk_gradient_permutation.png     : histogramme du Q2 nul vs Q2 observé
  - risk_gradient_projection.png      : risque observé vs position sur le
                                         gradient
  - permanova_permutation_<col>.png   : histogramme du pseudo-F nul vs observé
  Utiliser --no-plots pour désactiver la génération de figures.
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PALETTE = [
    "#FFA2A2A2", "#FFD230AB", "#35530EAC", "#5EE9B6A6", "#E1712BA9",
    "#8EC5FFA4", "#C4B4FFAC", "#024970A7", "#FFA1ADA9", "#E71A0BA9",
    "#53E9FDA6", "#5FA529B0", "#2D9967A7", "#2C93B8A6", "#155EFCA6",
    "#7E22FEA4", "#C71CDEAB", "#EC2540AC", "#CAD5E2A4", "#314158A7",
    "#020618A9", "#82181AA7", "#BBF451AE", "#F3A8FFAB", "#711378A9",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="PCoA + gradient de risque + PERMANOVA sur matrice de distance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--distance-matrix", required=True,
                   help="distance_matrix.tsv (patients x patients)")
    p.add_argument("--risk-file", default=None,
                   help="TSV 2 colonnes: patient, risk (score continu ou ordinal). ")
    p.add_argument("--metadata", default=None,
                   help="TSV avec colonne 'patient' (ou index) + colonnes de "
                        "métadonnées arbitraires (age, run, clinical_group, ...). "
                        "Utilisé pour --color-by et --permanova-column.")
    p.add_argument("--color-by", default=None,
                   help="Colonne de --metadata pour colorer le nuage PCoA. "
                        "Numérique -> dégradé continu (viridis). Non numérique "
                        "-> palette catégorielle (identique à clustering.py). "
                        "Si omis et --risk-file fourni, colore par le risque.")
    p.add_argument("--permanova-column", default=None,
                   help="Colonne CATÉGORIELLE de --metadata à tester par "
                        "PERMANOVA directement sur la matrice de distance "
                        "(ex. 'run' pour l'effet batch, 'clinical_group' pour "
                        "l'association clinique). Indépendant de --risk-file : "
                        "PERMANOVA teste une variable de groupe, pas un score "
                        "continu.")
    p.add_argument("--lingoes-correction", action="store_true", default=True,
                   help="Corrige les distances si des valeurs propres négatives "
                        "existent (défaut: activé).")
    p.add_argument("--no-lingoes-correction", dest="lingoes_correction", action="store_false")
    p.add_argument("--max-axes", type=int, default=None,
                   help="Nombre max d'axes PCoA considérés pour le gradient de "
                        "risque (défaut : min(10, n_patients // 3), pour limiter "
                        "le risque de surajustement sur une petite cohorte).")
    p.add_argument("--n-permutations", type=int, default=999,
                   help="Nombre de permutations pour le test du gradient ET "
                        "pour PERMANOVA (défaut : 999)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--tag", default=None, help="Tag optionnel pour les sorties (préfixe des fichiers).")
    p.add_argument("--outdir", "-o", required=True)
    p.add_argument("--no-plots", dest="make_plots", action="store_false", default=True,
                   help="Désactive la génération des figures PNG (TSV seuls).")
    p.add_argument("--label-bool", action="store_true", default=False,
                   help="Annote chaque point du nuage d'ordination avec l'ID patient ")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _suffix(tag):
    return f"_{tag}" if tag else ""


# --------------------------------------------------------------------------- #
# PCoA (Gower) avec diagnostic + correction optionnelle des valeurs propres négatives
# --------------------------------------------------------------------------- #
def _gower_decomposition(dist: np.ndarray):
    """Double-centrage de la matrice des distances au carré (méthode de Gower)."""
    n = dist.shape[0]
    D2 = dist ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2  # garde-fou : force la symétrie exacte (résidus d'arrondi flottant)
    eigvals, eigvecs = np.linalg.eigh(B)  # eigh : B symétrique, valeurs propres réelles
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def lingoes_correction(dist: np.ndarray) -> np.ndarray:
    """
    Correction de Lingoes : ajoute une constante additive aux distances
    au carré hors-diagonale pour éliminer les valeurs propres négatives de la
    décomposition de Gower, tout en préservant l'ordre relatif des distances
    originales.
    """
    eigvals, _ = _gower_decomposition(dist)
    lambda_min = eigvals.min()
    if lambda_min >= -1e-8:
        return dist  # rien à corriger
    c = -2 * lambda_min
    n = dist.shape[0]
    D2_corrected = dist ** 2 + c * (1 - np.eye(n))
    np.fill_diagonal(D2_corrected, 0.0)
    return np.sqrt(D2_corrected)


def pcoa(dist: np.ndarray, patients: list, apply_lingoes: bool):
    eigvals_raw, _ = _gower_decomposition(dist)
    total_abs = np.abs(eigvals_raw).sum()
    neg_mass = -eigvals_raw[eigvals_raw < 0].sum()
    frac_neg = float(neg_mass / total_abs) if total_abs > 0 else 0.0
    log.info("Masse propre négative avant correction : %.2f%% (diagnostic de non-euclidianité)",
              100 * frac_neg)

    dist_used = dist
    corrected = False
    if frac_neg > 1e-6 and apply_lingoes:
        log.info("Correction de Lingoes appliquée (valeurs propres négatives non négligeables).")
        dist_used = lingoes_correction(dist)
        corrected = True
    elif frac_neg > 0.10 and not apply_lingoes:
        log.warning(
            "%.1f%% de masse propre négative SANS correction appliquée : "
            "interpréter les axes de plus faible variance avec prudence.",
            100 * frac_neg,
        )

    eigvals, eigvecs = _gower_decomposition(dist_used)
    pos = eigvals > 1e-8
    coords = eigvecs[:, pos] * np.sqrt(eigvals[pos])

    total_var = eigvals[pos].sum()
    pct_var = eigvals[pos] / total_var * 100
    cum_var = np.cumsum(pct_var)

    eig_table = pd.DataFrame({
        "axis": [f"PCo{i+1}" for i in range(pos.sum())],
        "eigenvalue": eigvals[pos],
        "pct_variance": pct_var,
        "cumulative_pct_variance": cum_var,
    })
    coord_df = pd.DataFrame(
        coords, index=patients,
        columns=[f"PCo{i+1}" for i in range(coords.shape[1])],
    )
    return coord_df, eig_table, frac_neg, corrected


# --------------------------------------------------------------------------- #
# LOOCV pour la régression risque ~ axes PCoA (choix honnête de k)
# --------------------------------------------------------------------------- #
def _ols_fit_predict(X_train, y_train, x_test):
    Xd = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    beta, *_ = np.linalg.lstsq(Xd, y_train, rcond=None)
    xd_test = np.hstack([[1.0], x_test])
    return xd_test @ beta, beta


def loocv_q2(X: np.ndarray, y: np.ndarray) -> float:
    """
    Q2 leave-one-out : ré-estime le modèle en excluant chaque patient à tour
    de rôle et mesure la capacité prédictive hors-échantillon.
    """
    n = X.shape[0]
    preds = np.empty(n)
    idx_all = np.arange(n)
    for i in range(n):
        train = idx_all != i
        pred, _ = _ols_fit_predict(X[train], y[train], X[i])
        preds[i] = pred
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def best_k_by_loocv(coords: np.ndarray, risk: np.ndarray, max_axes: int):
    """Explore k=1..max_axes, retourne (k optimal, Q2 optimal, table complète)."""
    rows = []
    best_k, best_q2 = 1, -np.inf
    for k in range(1, max_axes + 1):
        q2 = loocv_q2(coords[:, :k], risk)
        rows.append({"k": k, "Q2_loocv": q2})
        if q2 > best_q2:
            best_k, best_q2 = k, q2
    table = pd.DataFrame(rows)
    return best_k, best_q2, table


# --------------------------------------------------------------------------- #
# Correction Benjamini-Hochberg (comparaisons multiples sur les axes)
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


# --------------------------------------------------------------------------- #
# PERMANOVA
# --------------------------------------------------------------------------- #
def permanova(dist: np.ndarray, groups: np.ndarray, n_permutations: int = 999,
              random_state: int = 42) -> dict:
    """
    PERMANOVA calculée directement sur la matrice de distance.
    """
    N = dist.shape[0]
    groups = np.asarray(groups)
    uniq_groups = np.unique(groups)
    a = len(uniq_groups)
    if a < 2:
        raise ValueError("PERMANOVA nécessite au moins 2 groupes distincts.")

    D2 = dist ** 2
    iu = np.triu_indices(N, k=1)
    ss_total = D2[iu].sum() / N

    def _ss_within(D2_mat, grp):
        ss_w = 0.0
        for g in uniq_groups:
            idx = np.where(grp == g)[0]
            n_g = len(idx)
            if n_g < 2:
                continue
            sub = D2_mat[np.ix_(idx, idx)]
            iu_g = np.triu_indices(n_g, k=1)
            ss_w += sub[iu_g].sum() / n_g
        return ss_w

    ss_within = _ss_within(D2, groups)
    ss_between = ss_total - ss_within

    df_between = a - 1
    df_within = N - a
    if df_within <= 0:
        raise ValueError(
            f"PERMANOVA : degrés de liberté intra-groupe <= 0 (N={N}, a={a}). "
            "Trop de groupes pour le nombre de patients disponibles."
        )

    f_obs = (ss_between / df_between) / (ss_within / df_within)

    rng = np.random.RandomState(random_state)
    f_null = np.empty(n_permutations)
    for b in range(n_permutations):
        perm_groups = rng.permutation(groups)
        ss_w_perm = _ss_within(D2, perm_groups)
        ss_b_perm = ss_total - ss_w_perm
        f_null[b] = (ss_b_perm / df_between) / (ss_w_perm / df_within)

    p_value = (np.sum(f_null >= f_obs) + 1) / (n_permutations + 1)

    return {
        "a_groups": a,
        "N": N,
        "df_between": df_between,
        "df_within": df_within,
        "SS_total": ss_total,
        "SS_between": ss_between,
        "SS_within": ss_within,
        "pseudo_F": f_obs,
        "n_permutations": n_permutations,
        "null_F_median": float(np.median(f_null)),
        "null_F_95th_percentile": float(np.percentile(f_null, 95)),
        "p_value": p_value,
    }, f_null


def plot_permanova(f_null: np.ndarray, f_obs: float, p_value: float,
                    column: str, outpath: str, dpi: int = 150):
    """Distribution nulle du pseudo-F (labels permutés) vs pseudo-F observé."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(f_null, bins=40, color="#8C8C8C", alpha=0.8, label="Pseudo-F (labels permutés)")
    ax.axvline(f_obs, color="#C44E52", linewidth=2, label=f"Pseudo-F observé = {f_obs:.3f}")
    ax.set_xlabel("Pseudo-F")
    ax.set_ylabel("Nombre de permutations")
    ax.set_title(f"PERMANOVA — '{column}' (p = {p_value:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Figure écrite : %s", outpath)


# --------------------------------------------------------------------------- #
# Coloration générique par métadonnées (cohérente avec clustering.py)
# --------------------------------------------------------------------------- #
def _load_metadata(path, patients: list):
    if path is None:
        return None
    meta = pd.read_csv(path, sep="\t")
    # Accepte soit une colonne "patient", soit un index déjà nommé.
    if "patient" in meta.columns:
        meta = meta.set_index("patient")
    else:
        meta = meta.set_index(meta.columns[0])
    missing = set(patients) - set(meta.index)
    if missing:
        log.warning(
            "Métadonnées absentes pour %d patients : %s",
            len(missing), ", ".join(sorted(missing)[:5]),
        )
    return meta.reindex(patients)


def _color_by_column(meta: pd.DataFrame, patients: list, color_by: str):
    """
    Coloration par métadonnée (colonne de --metadata) : continue (viridis) ou catégorielle (palette fixe).
    """
    if meta is None or color_by not in meta.columns:
        raise ValueError(f"Colonne '{color_by}' absente des métadonnées fournies.")

    raw = meta.loc[patients, color_by]
    numeric = pd.to_numeric(raw, errors="coerce")

    if numeric.notna().all():
        vmin, vmax = float(numeric.min()), float(numeric.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        colors = [plt.cm.viridis(norm(v)) for v in numeric]
        log.info("Coloration continue par '%s' (viridis, min=%.3f, max=%.3f)",
                  color_by, vmin, vmax)
        return colors, "continuous", norm, color_by

    groups = raw.fillna("N/A").astype(str).tolist()
    uniq = sorted(set(groups))
    if len(uniq) > len(PALETTE):
        log.warning(
            "  %d catégories dans '%s' > %d couleurs disponibles dans la "
            "palette — certaines catégories partageront la même couleur.",
            len(uniq), color_by, len(PALETTE),
        )
    cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(uniq)}
    colors = [cmap[g] for g in groups]
    patches = [mpatches.Patch(color=cmap[g], label=g) for g in uniq]
    log.info("Coloration catégorielle par '%s' : %d groupes détectés", color_by, len(uniq))
    return colors, "categorical", patches, color_by


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_scree(eig_table: pd.DataFrame, outpath: str, dpi: int = 150, n_show: int = 15):
    """Variance expliquée (barres) + cumulée (ligne) par axe PCoA."""
    sub = eig_table.iloc[:n_show]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(sub["axis"], sub["pct_variance"], color="#4C72B0", alpha=0.85)
    ax1.set_xlabel("Axe PCoA")
    ax1.set_ylabel("Variance expliquée (%)", color="#4C72B0")
    ax1.tick_params(axis="x", rotation=90)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(sub["axis"], sub["cumulative_pct_variance"], color="#DD8452",
              marker="o", linewidth=1.5)
    ax2.set_ylabel("Variance cumulée (%)", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax2.set_ylim(0, 100)

    ax1.set_title("PCoA — variance expliquée par axe")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Figure écrite : %s", outpath)


def plot_ordination(coord_df: pd.DataFrame, eig_table: pd.DataFrame, outpath: str,
                     colors=None, mode=None, legend_info=None, colorbar_label=None,
                     label_bool: bool = False, dpi: int = 150):
    """
    Nuage de points des patients dans l'espace PCoA (PCo1 x PCo2, et PCo1 x PCo3
    si disponible).
    """
    def pct(axis_name):
        row = eig_table.loc[eig_table["axis"] == axis_name, "pct_variance"]
        return float(row.iloc[0]) if len(row) else float("nan")

    n_axes = coord_df.shape[1]
    n_panels = 2 if n_axes >= 3 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)
    axes = axes[0]

    panels = [("PCo1", "PCo2")]
    if n_axes >= 3:
        panels.append(("PCo1", "PCo3"))

    for ax, (xax, yax) in zip(axes, panels):
        x = coord_df[xax].to_numpy()
        y = coord_df[yax].to_numpy()
        c = colors if colors is not None else "#4C72B0"
        ax.scatter(x, y, c=c, s=55, edgecolor="white", linewidth=0.6)
        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
        ax.axvline(0, color="grey", linewidth=0.5, zorder=0)
        ax.set_xlabel(f"{xax} ({pct(xax):.1f}% var.)")
        ax.set_ylabel(f"{yax} ({pct(yax):.1f}% var.)")
        if label_bool:
            for pid, xi, yi in zip(coord_df.index, x, y):
                ax.annotate(str(pid), (xi, yi), fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")

    if mode == "continuous" and legend_info is not None:
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=legend_info)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes), shrink=0.85, pad=0.02)
        cbar.set_label(colorbar_label or "Valeur")
    elif mode == "categorical" and legend_info:
        fig.legend(handles=legend_info, loc="upper right", fontsize=9,
                   title=colorbar_label, framealpha=0.85)

    fig.suptitle("Ordination PCoA — relations entre patients")
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure écrite : %s", outpath)


def plot_permutation(null_q2: np.ndarray, observed_q2: float, p_value: float,
                      outpath: str, dpi: int = 150):
    """Distribution nulle du Q2 (risque permuté) vs Q2 observé."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(null_q2, bins=40, color="#8C8C8C", alpha=0.8, label="Q2 (risque permuté)")
    ax.axvline(observed_q2, color="#C44E52", linewidth=2,
               label=f"Q2 observé = {observed_q2:.3f}")
    ax.set_xlabel("Q2 (LOOCV)")
    ax.set_ylabel("Nombre de permutations")
    ax.set_title(f"Test de permutation du gradient de risque (p = {p_value:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Figure écrite : %s", outpath)


def plot_gradient_projection(scores_df: pd.DataFrame, outpath: str, dpi: int = 150):
    """Risque observé vs position sur le gradient, avec la droite de tendance."""
    x = scores_df["risk_observed"].to_numpy()
    y = scores_df["risk_gradient_projection"].to_numpy()
    rho, pval = spearmanr(x, y)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, c="#4C72B0", s=55, edgecolor="white", linewidth=0.6)
    order = np.argsort(x)
    z = np.polyfit(x, y, 1)
    ax.plot(x[order], np.polyval(z, x[order]), color="#C44E52", linewidth=1.5,
            linestyle="--")
    ax.set_xlabel("Risque observé")
    ax.set_ylabel("Position sur le gradient de risque (projection PCoA)")
    ax.set_title(f"Gradient de risque — Spearman ρ = {rho:.2f} (p = {pval:.4f})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    log.info("Figure écrite : %s", outpath)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dist_df = pd.read_csv(args.distance_matrix, sep="\t", index_col=0)
    dist_df = dist_df.loc[dist_df.index, dist_df.index]  # garantit alignement lignes/colonnes
    patients = dist_df.index.tolist()
    dist = dist_df.to_numpy(dtype=np.float64)
    log.info("Matrice de distance chargée : %d patients", len(patients))

    coord_df, eig_table, frac_neg, corrected = pcoa(dist, patients, args.lingoes_correction)
    eig_table.to_csv(f"{args.outdir}/pcoa_eigenvalues{_suffix(args.tag)}.tsv", sep="\t", index=False)
    coord_df.to_csv(f"{args.outdir}/pcoa_coordinates{_suffix(args.tag)}.tsv", sep="\t")
    log.info("PCoA écrite. Variance cumulée des 5 premiers axes : %.1f%%",
              eig_table["cumulative_pct_variance"].iloc[:5].iloc[-1] if len(eig_table) >= 5
              else eig_table["cumulative_pct_variance"].iloc[-1])

    if args.make_plots:
        plot_scree(eig_table, f"{args.outdir}/pcoa_scree{_suffix(args.tag)}.png", dpi=args.dpi)

    # ── Métadonnées (color-by + permanova) ────────────────────────────
    meta = _load_metadata(args.metadata, patients)

    # ── PERMANOVA (indépendant du gradient) ───────────────────────────
    if args.permanova_column is not None:
        if meta is None or args.permanova_column not in meta.columns:
            log.error(
                "--permanova-column='%s' demandé mais absent de --metadata. "
                "PERMANOVA ignorée.", args.permanova_column,
            )
        else:
            groups_raw = meta.loc[patients, args.permanova_column]
            valid = groups_raw.notna()
            if not valid.all():
                log.warning(
                    "%d patients sans valeur pour '%s' — exclus de PERMANOVA.",
                    int((~valid).sum()), args.permanova_column,
                )
            groups_valid = groups_raw[valid].astype(str).to_numpy()
            idx_valid = np.where(valid.to_numpy())[0]
            dist_valid = dist[np.ix_(idx_valid, idx_valid)]

            try:
                result, f_null = permanova(
                    dist_valid, groups_valid,
                    n_permutations=args.n_permutations,
                    random_state=args.random_state,
                )
                result_df = pd.DataFrame([result])
                out_path = f"{args.outdir}/permanova_{args.permanova_column}{_suffix(args.tag)}.tsv"
                result_df.to_csv(out_path, sep="\t", index=False)
                log.info(
                    "PERMANOVA '%s' : pseudo-F=%.4f (a=%d groupes, N=%d), p=%.4f -> %s",
                    args.permanova_column, result["pseudo_F"], result["a_groups"],
                    result["N"], result["p_value"], out_path,
                )
                if result["p_value"] >= 0.05:
                    log.info(
                        "  p=%.4f >= 0.05 : pas de différence significative entre "
                        "groupes de '%s' détectée sur les profils de méthylation.",
                        result["p_value"], args.permanova_column,
                    )
                else:
                    log.warning(
                        "  p=%.4f < 0.05 : différence significative détectée entre "
                        "groupes de '%s'. Vérifier qu'il ne s'agit pas d'un effet "
                        "de dispersion intra-groupe plutôt qu'une vraie séparation "
                        "de centroïdes (cf. limite PERMDISP dans la docstring).",
                        result["p_value"], args.permanova_column,
                    )
                if args.make_plots:
                    plot_permanova(
                        f_null, result["pseudo_F"], result["p_value"],
                        args.permanova_column,
                        f"{args.outdir}/permanova_permutation_{args.permanova_column}{_suffix(args.tag)}.png",
                        dpi=args.dpi,
                    )
            except ValueError as e:
                log.error("PERMANOVA impossible pour '%s' : %s", args.permanova_column, e)

    # ── Résolution de la coloration du nuage PCoA ─────────────────────
    # Priorité : --color-by (métadonnées génériques) > --risk-file (rétro-
    # compatibilité) > pas de coloration.
    colors = mode = legend_info = colorbar_label = None
    risk = risk_series = None
    risk_continuous = False

    if args.color_by is not None:
        if meta is None:
            log.error("--color-by='%s' demandé mais --metadata absent. Ignoré.", args.color_by)
        else:
            colors, mode, legend_info, colorbar_label = _color_by_column(meta, patients, args.color_by)

    if args.risk_file is not None:
        risk_df = pd.read_csv(args.risk_file, sep=r"\s+", engine="python", header=0)
        if not {"patient", "risk"}.issubset(risk_df.columns):
            raise ValueError("--risk-file doit contenir les colonnes 'patient' et 'risk'")
        risk_df = risk_df.set_index("patient")

        risk_raw = risk_df["risk"]
        risk_numeric = pd.to_numeric(risk_raw, errors="coerce")

        if risk_numeric.notna().all():
            risk_continuous = True
            risk_series = risk_numeric
        else:
            if risk_numeric.notna().any():
                log.warning(
                    "Mélange de valeurs numériques et non numériques dans 'risk'. "
                    "Traité comme catégoriel."
                )
            missing_risk_rows = risk_raw[risk_raw.isna()].index.tolist()
            if missing_risk_rows:
                log.warning("Suppression de %d patient(s) sans valeur 'risk' : %s",
                            len(missing_risk_rows), missing_risk_rows)
            risk_series = risk_raw.astype(str)
            risk_continuous = False

        # Patients communs entre matrice de distance et fichier de risque
        patients_in_risk = set(risk_series.index)
        missing_in_riskfile = [p for p in patients if p not in patients_in_risk]
        if missing_in_riskfile:
            log.warning(
                "%d patient(s) de la matrice de distance absents du fichier de "
                "risque : retirés de l'analyse : %s",
                len(missing_in_riskfile), missing_in_riskfile,
            )
            keep = [p for p in patients if p in patients_in_risk]
            dist_df = dist_df.loc[keep, keep]
            patients = dist_df.index.tolist()
            dist = dist_df.to_numpy(dtype=np.float64)
            coord_df = coord_df.loc[patients]
            coord_df.to_csv(f"{args.outdir}/pcoa_coordinates{_suffix(args.tag)}.tsv", sep="\t")

        extra_in_risk = sorted(set(risk_series.index) - set(patients))
        if extra_in_risk:
            log.warning("%d patient(s) du fichier de risque absents de la matrice de "
                        "distance, ignorés : %s", len(extra_in_risk), extra_in_risk)

        risk_series = risk_series.loc[patients]
        if risk_continuous:
            risk = risk_series.to_numpy(dtype=np.float64)
        else:
            risk = pd.Categorical(risk_series).codes.astype(np.float64)

        # Coloration de repli par le risque, uniquement si --color-by n'a
        # rien produit (compatibilité avec l'usage historique du script).
        if colors is None:
            if risk_continuous:
                vmin, vmax = float(risk_series.min()), float(risk_series.max())
                norm = Normalize(vmin=vmin, vmax=vmax)
                colors = [plt.cm.viridis(norm(v)) for v in risk_series]
                mode, legend_info, colorbar_label = "continuous", norm, "Risque"
            else:
                uniq = sorted(set(risk_series))
                cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(uniq)}
                colors = [cmap[g] for g in risk_series]
                legend_info = [mpatches.Patch(color=cmap[g], label=g) for g in uniq]
                mode, colorbar_label = "categorical", "Risque"

    if args.make_plots:
        plot_ordination(
            coord_df, eig_table, f"{args.outdir}/pcoa_ordination{_suffix(args.tag)}.png",
            colors=colors, mode=mode, legend_info=legend_info, colorbar_label=colorbar_label,
            label_bool=args.label_bool, dpi=args.dpi,
        )

    # ── Analyse de gradient (uniquement si risque continu disponible) ──
    if args.risk_file is None:
        log.info("Pas de --risk-file fourni : pas d'analyse de gradient.")
        return
    if not risk_continuous:
        log.info(
            "La colonne 'risk' contient des valeurs non numériques : la PCoA "
            "est tracée par catégories, mais l'analyse de gradient (régression "
            "continue) est interrompue. Pour tester une variable catégorielle, "
            "utiliser --permanova-column à la place."
        )
        return

    n_patients = len(patients)
    max_axes = args.max_axes or max(1, min(10, n_patients // 3))
    max_axes = min(max_axes, coord_df.shape[1], n_patients - 2)
    log.info("Exploration de k=1 à %d axes pour le gradient de risque (n_patients=%d)",
             max_axes, n_patients)

    rows = []
    for i in range(max_axes):
        axis_vals = coord_df.iloc[:, i].to_numpy()
        rho, pval = spearmanr(axis_vals, risk)
        rows.append({"axis": coord_df.columns[i], "spearman_rho": rho, "p_value": pval})
    corr_table = pd.DataFrame(rows)
    corr_table["p_adj_BH"] = benjamini_hochberg(corr_table["p_value"].to_numpy())
    corr_table.to_csv(f"{args.outdir}/axis_risk_correlation{_suffix(args.tag)}.tsv", sep="\t", index=False)
    log.info("Corrélations axe~risque écrites (BH-ajustées, %d tests)", max_axes)

    coords_arr = coord_df.iloc[:, :max_axes].to_numpy()
    best_k, best_q2, loocv_table = best_k_by_loocv(coords_arr, risk, max_axes)
    loocv_table.to_csv(f"{args.outdir}/risk_gradient_loocv{_suffix(args.tag)}.tsv", sep="\t", index=False)
    log.info("k retenu par LOOCV : %d (Q2=%.4f)", best_k, best_q2)
    if best_q2 < 0:
        log.warning(
            "Q2 LOOCV négatif au meilleur k : le modèle prédit moins bien que la "
            "moyenne du risque observé. Aucun gradient de risque fiable détectable."
        )

    rng = np.random.RandomState(args.random_state)
    null_q2 = np.empty(args.n_permutations)
    for b in range(args.n_permutations):
        risk_perm = rng.permutation(risk)
        _, q2_perm, _ = best_k_by_loocv(coords_arr, risk_perm, max_axes)
        null_q2[b] = q2_perm
    p_value = (np.sum(null_q2 >= best_q2) + 1) / (args.n_permutations + 1)
    pd.DataFrame({
        "observed_Q2": [best_q2],
        "observed_k": [best_k],
        "n_permutations": [args.n_permutations],
        "null_Q2_median": [np.median(null_q2)],
        "null_Q2_95th_percentile": [np.percentile(null_q2, 95)],
        "p_value": [p_value],
    }).to_csv(f"{args.outdir}/permutation_test{_suffix(args.tag)}.tsv", sep="\t", index=False)
    log.info(
        "Test de permutation (n=%d) : Q2 observé=%.4f, médiane nulle=%.4f, p=%.4f",
        args.n_permutations, best_q2, np.median(null_q2), p_value,
    )
    if p_value >= 0.05:
        log.warning(
            "p=%.4f >= 0.05 : le gradient de risque observé n'est PAS distinguable "
            "de ce qu'on obtiendrait avec un risque assigné au hasard.", p_value,
        )

    if args.make_plots:
        plot_permutation(null_q2, best_q2, p_value,
                          f"{args.outdir}/risk_gradient_permutation{_suffix(args.tag)}.png", dpi=args.dpi)

    Xd = np.hstack([np.ones((n_patients, 1)), coords_arr[:, :best_k]])
    beta, *_ = np.linalg.lstsq(Xd, risk, rcond=None)
    weights = beta[1:]
    fitted = Xd @ beta
    weights_norm = weights / np.linalg.norm(weights)
    raw_projection = coords_arr[:, :best_k] @ weights_norm

    direction_df = pd.DataFrame({
        "axis": coord_df.columns[:best_k],
        "weight_regression": weights,
        "weight_normalized": weights_norm,
    })
    direction_df.to_csv(f"{args.outdir}/risk_gradient_direction{_suffix(args.tag)}.tsv", sep="\t", index=False)

    scores_df = pd.DataFrame({
        "patient": patients,
        "risk_observed": risk,
        "risk_predicted_insample": fitted,
        "risk_gradient_projection": raw_projection,
    }).sort_values("risk_gradient_projection")
    scores_df.to_csv(f"{args.outdir}/risk_gradient_scores{_suffix(args.tag)}.tsv", sep="\t", index=False)

    if args.make_plots:
        plot_gradient_projection(scores_df, f"{args.outdir}/risk_gradient_projection{_suffix(args.tag)}.png",
                                  dpi=args.dpi)

    log.info("Gradient de risque écrit — patients triés par position le long du gradient.")
    log.info("Terminé. Résultats dans %s", args.outdir)


if __name__ == "__main__":
    main()