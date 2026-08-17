#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clustering.py — Analyse non-supervisée des profils de méthylation
==================================================================

Prend en entrée methylation_cohort.tsv produit par le pipeline
(une ligne par site × mod_code, colonnes beta_<sample> / cov_<sample>).

Sous-commandes :

  1. prepare
        methylation_cohort.tsv → clustering_matrix.parquet (défaut) ou .tsv
        - Filtre sur mod_code (m ou h)
        - Filtre de complétude par site (réduit la dépendance à
          l'imputation, distinct du filtre semi-strict amont)
        - Imputation des NaN résiduels par médiane / kNN / PCA itérative
        - Sélection optionnelle des top N sites les plus variables
          (variance inter-patients, calculée après imputation —
          déconseillée pour les cohortes modérées, cf. --top-n)
        - Sortie : matrice patients × sites, prête pour sklearn.
          Format Parquet (zstd, float32) recommandé pour les grandes cohortes
          (>>10× plus rapide et plus compact que TSV).

  2. run_pca
        clustering_matrix.parquet → pca_coords.tsv + pca_variance.tsv
                                   + pca_scatter.png + pca_screeplot.png

  3. run_umap
        clustering_matrix.parquet → umap_coords.tsv + umap_scatter.png

  4. run_kmeans
        clustering_matrix.parquet → kmeans_labels.tsv + kmeans_inertia.tsv
        + kmeans_elbow.png + pca_scatter_kmeans.png + umap_scatter_kmeans.png

  5. run_gmm
        clustering_matrix.parquet → gmm_labels.tsv + gmm_bic.tsv
        + gmm_scatter.png + pca_scatter_gmm.png + umap_scatter_gmm.png

  6. run_available_case
        clustering_matrix.parquet (avec NaN) → distance_matrix.tsv
        + dendrogram.png + hierarchical_labels.tsv + kmedoids_labels.tsv

Métadonnées cliniques (optionnel) :
    --metadata : TSV avec colonne "sample" + colonnes de groupes/phénotypes
    Quand fourni, les plots colorent par groupe au lieu de par cluster.

Format de la matrice (Parquet) :
    Stockée en orientation sites × patients (millions de lignes, ~40-100
    colonnes) pour optimiser le format colonnaire Parquet. Transposée en
    mémoire en patients × sites (attendu par sklearn) à chaque chargement.
    Les valeurs sont en float32 (7 décimales significatives — largement
    suffisant pour des beta values stockées à 4 décimales).
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture 
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
import umap

try:
    from adjustText import adjust_text
    _HAS_ADJUSTTEXT = True
except ImportError:
    _HAS_ADJUSTTEXT = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PALETTE = [
    "#FFA2A2A2", "#FFD230AB", "#35530EAC", "#5EE9B6A6", "#E1712BA9",
    "#8EC5FFA4", "#C4B4FFAC", "#024970A7", "#FFA1ADA9", "#E71A0BA9",
    "#53E9FDA6", "#5FA529B0", "#2D9967A7", "#2C93B8A6", "#155EFCA6",
    "#7E22FEA4", "#C71CDEAB", "#EC2540AC", "#CAD5E2A4", "#314158A7",
    "#020618A9", "#82181AA7", "#BBF451AE", "#F3A8FFAB", "#711378A9",
    
]


# ========================= ARGPARSE ======================

def parse_args():
    p = argparse.ArgumentParser(
        description="Clustering méthylation — PCA / UMAP / k-means",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── 1. prepare ────────────────────────────────────────
    pr = sub.add_parser("prepare",
        help="methylation_cohort.tsv → clustering_matrix.tsv")
    pr.add_argument("--cohort-tsv", required=True,
                    help="methylation_cohort.tsv produit par export_tsv")
    pr.add_argument("--mod-code", default="m", choices=["m", "h"],
                    help="Code de modification à utiliser (défaut : m = 5mC)")
    pr.add_argument("--top-n", type=int, default=None,
                    help="Nombre de sites les plus variables à conserver (défaut : tous). "
                         "La variance est calculée après imputation, donc déconseillé pour les cohortes modérées.")
    pr.add_argument("--min-completeness", type=float, default=0.7,
                    help="Fraction minimale de patients non-NaN requise par site pour être conservé (défaut : 0.7). Filtre "
                         "appliqué après le filtre semi-strict amont (déjà présent dans methylation_cohort.tsv), spécifiquement "
                         "pour limiter le recours à l'imputation en amont du clustering. ATTENTION : avec un grand nombre de "
                         "patients, cet effet de seuil peut être très abrupt (cf. --completeness-histogram pour diagnostiquer "
                         "avant de fixer ce seuil).")
    pr.add_argument("--completeness-histogram", default=None,
                    help="Chemin de sortie pour un histogramme (TSV, comptage par tranche de 5%%) de la complétude par site, "
                         "calculé avant filtrage. Recommandé sur les grandes cohortes pour choisir --min-completeness de façon "
                         "data-driven plutôt qu'arbitraire (ex. repérer une distribution bimodale en adaptive sampling et placer "
                         "le seuil dans la vallée entre les deux modes).")
    pr.add_argument("--imputation-method", default="pca",
                    choices=["median", "knn", "pca", "random", "no_impute"],
                    help="Méthode d'imputation des NaN résiduels (défaut : pca)."
                         "'median' : médiane du site sur les patients "
                         "couverts (legacy — traite chaque site indépendamment et écrase la covariance inter-sites)."
                         "'knn' : k plus proches voisins entre patients (similarité de profil global)."
                         "'pca' : PCA itérative / EM (préserve la structure de covariance inter-sites — recommandé pour la suite PCA/k-means)."
                         "'random' : imputation aléatoire (utile pour tester la robustesse du clustering)."
                         "'no_impute' : pas d'imputation, conserve les NaN (utile pour run_available_case).")
    pr.add_argument("--knn-neighbors", type=int, default=5,
                    help="Nombre de voisins pour --imputation-method knn (défaut : 5)")
    pr.add_argument("--pca-n-components", type=int, default=10,
                    help="Nombre de composantes pour --imputation-method pca (défaut : 10)")
    pr.add_argument("--pca-max-iter", type=int, default=50,
                    help="Itérations maximales pour la convergence de l'imputation PCA (défaut : 50)")
    pr.add_argument("--pca-tol", type=float, default=1e-4,
                    help="Seuil de convergence — variation relative de la norme de Frobenius entre deux itérations (défaut : 1e-4)")
    pr.add_argument("--output-format", default="parquet",
                    choices=["parquet", "tsv"],
                    help="Format de la matrice de sortie (défaut : parquet). ")
    pr.add_argument("--exclude-samples", nargs="+", default=None,
                    help="Liste de patients à exclure du clustering (sample names)")
    pr.add_argument("--seuil", default=False,
                   help="Appliquer le seuil de methylation 0.5 (défaut : False)")
    pr.add_argument("--tag", default=None,
                    help="Suffixe de sortie ajouté aux fichiers générés")
    pr.add_argument("--outdir", "-o", required=True,
                    help="Répertoire où sont enregistrées les fichiers de sortie")

    # ── 2. run_pca ────────────────────────────────────────
    pca_p = sub.add_parser("run_pca",
        help="clustering_matrix.tsv → coordonnées PCA + plots")
    pca_p.add_argument("--matrix", required=True)
    pca_p.add_argument("--n-components", type=int, default=20)
    pca_p.add_argument("--use-robust-scaler", action="store_true",
                       help="Utiliser RobustScaler (median/IQR) au lieu de StandardScaler. "
                            "Recommande avec tous les sites ou en presence d'outliers.")
    pca_p.add_argument("--metadata", default=None)
    pca_p.add_argument("--color-by", default=None)
    pca_p.add_argument("--label-bool", default=True,
                       choices=["True", "False"],
                       help="Booléen qui indique si les étiquettes sont présentes sur les figures")
    pca_p.add_argument("--tag", default=None,
                       help="Suffixe de sortie ajouté aux fichiers générés")
    pca_p.add_argument("--outdir", "-o", required=True,
                    help="Répertoire où sont enregistrées les fichiers de sortie")

    # ── 3. run_umap ───────────────────────────────────────
    um = sub.add_parser("run_umap",
        help="clustering_matrix.tsv → coordonnées UMAP + plot")
    um.add_argument("--matrix", required=True)
    um.add_argument("--n-neighbors", type=int, default=10)
    um.add_argument("--min-dist", type=float, default=0.1)
    um.add_argument("--use-robust-scaler", action="store_true",
                       help="Utiliser RobustScaler au lieu de StandardScaler.")
    um.add_argument("--random-state", type=int, default=42)
    um.add_argument("--metadata", default=None)
    um.add_argument("--color-by", default=None)
    um.add_argument("--label-bool", default=True,
                    choices=["True", "False"],
                    help="Booléen qui indique si les étiquettes sont présentes sur les figures")
    um.add_argument("--tag", default=None,
                    help="Suffixe de sortie ajouté aux fichiers générés")
    um.add_argument("--outdir", "-o", required=True,
                    help="Répertoire où sont enregistrées les fichiers de sortie")

    # ── 4. run_kmeans ─────────────────────────────────────
    km = sub.add_parser("run_kmeans",
        help="clustering_matrix.tsv → labels k-means + plots")
    km.add_argument("--matrix", required=True)
    km.add_argument("--k-min", type=int, default=2)
    km.add_argument("--k-max", type=int, default=10)
    km.add_argument("--k-final", type=int, default=None,
                    help="K final (défaut : automatique via elbow)")
    km.add_argument("--pca-coords", default=None)
    km.add_argument("--umap-coords", default=None)
    km.add_argument("--random-state", type=int, default=42)
    km.add_argument("--use-robust-scaler", action="store_true",
                       help="Utiliser RobustScaler au lieu de StandardScaler.")
    km.add_argument("--label-bool", default=True,
                    choices=["True", "False"],
                    help="Booléen qui indique si les étiquettes sont présentes sur les figures")
    km.add_argument("--tag", default=None,
                    help="Suffixe de sortie ajouté aux fichiers générés")
    km.add_argument("--outdir", "-o", required=True,
                    help="Répertoire où sont enregistrées les fichiers de sortie")

    # ── 5. run_gmm ─────────────────────────────────────
    gmm = sub.add_parser("run_gmm",
        help="clustering_matrix.tsv → labels GMM + plots")
    gmm.add_argument("--matrix", required=True)
    gmm.add_argument("--k-min", type=int, default=2)
    gmm.add_argument("--k-max", type=int, default=10)
    gmm.add_argument("--k-final", type=int, default=None,
                    help="K final (défaut : automatique via BIC)")
    gmm.add_argument("--pca-coords", default=None)
    gmm.add_argument("--umap-coords", default=None)
    gmm.add_argument("--random-state", type=int, default=42)
    gmm.add_argument("--covariance-type", default="diag",
                    choices=["full", "tied", "diag", "spherical"],
                    help="Type de covariance du GMM. "
                         "'full' : matrice p×p pleine par composante — nécessite n_patients_par_cluster > p pour être "
                         "non-singulière ; avec p (sites) >> n (patients), la covariance est structurellement singulière "
                         "quel que soit reg_covar (ce n'est pas une question de réglage numérique). "
                         "'diag' : variances indépendantes par site, k*p paramètres, reste "
                         "identifiable même pour p grand — recommandé si l'entrée est la matrice de sites complète. "
                         "'spherical' : variance isotrope par composante, le plus contraint, robuste aux petits effectifs par "
                         "cluster (utile si singletons attendus). "
                         "'tied' : covariance partagée entre composantes. "
                         "'full'/'tied' ne sont défendables que sur un espace réduit (ex. coordonnées PCA/PCoA, p<n).")
    gmm.add_argument("--reg-covar", type=float, default=1e-6,
                    help="Terme additif sur la diagonale de covariance pour la stabilité "
                         "numérique (défaut sklearn : 1e-6). X étant toujours standardisé "
                         "par _get_scaler avant le fit (échelle ~1), le défaut est "
                         "généralement adapté ; à augmenter seulement en cas de warning "
                         "de dégénérescence explicite malgré covariance_type approprié.")
    gmm.add_argument("--use-robust-scaler", action="store_true",
                       help="Utiliser RobustScaler au lieu de StandardScaler.")
    gmm.add_argument("--label-bool", default=True,
                    choices=["True", "False"],
                    help="Booléen qui indique si les étiquettes sont présentes sur les figures")
    gmm.add_argument("--tag", default=None,
                    help="Suffixe de sortie ajouté aux fichiers générés")
    gmm.add_argument("--outdir", "-o", required=True,
                    help="Répertoire où sont enregistrées les fichiers de sortie")

    # ── 6. run_pairwise ─────────────────────────────
    ac = sub.add_parser("run_pairwise",
        help="Distances sur sites co-observés "
             "→ hiérarchique (average/complete linkage) + k-medoids. ")
    ac.add_argument("--matrix", required=True,
                    help="clustering_matrix.tsv produit par prepare (idéalement sans NaN).")
    ac.add_argument("--distance-metric", default="pearson",
                    choices=["pearson", "spearman", "euclidean"],
                    help="Distance à utiliser pour le clustering available-case. "
                         "'pearson' (par défaut) : 1 - corrélation de Pearson entre profils de patients. "
                         "'spearman' : 1 - corrélation de rangs de Spearman. "
                         "'euclidean' : distance euclidienne classique.")
    ac.add_argument("--linkage", default="average",
                    choices=["average", "complete"],
                    help="Méthode de linkage pour le clustering hiérarchique (défaut : average)."
                         "'average' (UPGMA) : distance entre clusters = moyenne des distances inter-paires — "
                         "valide pour toute dissimilarité, robuste aux outliers. "
                         "'complete' : distance = maximum des paires — tend à former des clusters compacts et sphériques, sensible aux outliers.")
    ac.add_argument("--k-min", type=int, default=2,
                    help="K minimum pour k-medoids (défaut : 2)")
    ac.add_argument("--k-max", type=int, default=10,
                    help="K maximum pour k-medoids (défaut : 10)")
    ac.add_argument("--k-final", type=int, default=None,
                    help="K final pour k-medoids. Si omis, sélection automatique par silhouette maximale.")
    ac.add_argument("--n-cut", type=int, default=None,
                    help="Couper le dendrogramme hiérarchique en N clusters. "
                         "Si omis, le dendrogramme complet est produit sans partition forcée.")
    ac.add_argument("--min-coobs", type=int, default=1000,
                    help="Nombre minimal de sites co-observés requis entre deux patients pour que leur distance soit considérée fiable. "
                         "Cette option est pertinente uniquement si la matrice contient des NaN. "
                         "Si la matrice est complète, ce paramètre est ignoré.")
    ac.add_argument("--pca-coords", default=None,
                    help="pca_coords.tsv (optionnel) : projeter les clusters sur le scatter PCA pour comparaison visuelle")
    ac.add_argument("--umap-coords", default=None,
                    help="umap_coords.tsv (optionnel) : même chose pour UMAP")
    ac.add_argument("--metadata", default=None)
    ac.add_argument("--color-by", default=None)
    ac.add_argument("--label-bool", default=True,
                    choices=["True", "False"],
                    help="Booléen qui indique si les étiquettes sont présentes sur les figures")
    ac.add_argument("--tag", default=None,
                    help="Suffixe de sortie ajouté aux fichiers générés")
    ac.add_argument("--outdir", "-o", required=True,
                    help="Répertoire où sont enregistrées les fichiers de sortie")

    return p.parse_args()


# ========================= HELPERS =======================

class _EpsilonRobustScaler(RobustScaler):
    """
    Variante de RobustScaler qui remplace les scales nulles ou quasi nulles
    par un epsilon explicite, afin d'éviter les divisions par zéro lorsque
    l'IQR d'un site est nul (ou presque) sur de très grandes matrices.
    """
    def __init__(self, *, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def fit(self, X, y=None):
        super().fit(X, y)
        self.scale_ = np.maximum(self.scale_, self.epsilon)
        return self


def _get_scaler(use_robust: bool):
    """
    Retourne le scaler a utiliser.

    StandardScaler : mean/std. Sensible aux outliers.
    RobustScaler   : median/IQR. Robuste aux outliers et aux sites extremes.

    Recommande : RobustScaler si --use-robust-scaler ou si tous les sites (~2M).
    """
    if use_robust:
        log.info("Utilisation de RobustScaler (median/IQR, robuste aux outliers, epsilon=1e-8)")
        return _EpsilonRobustScaler(quantile_range=(10, 90), epsilon=1e-8) # return RobustScaler(quantile_range=(10, 90))  # robuste aux outliers
    else:
        log.info("Utilisation de StandardScaler (mean/std)")
        return StandardScaler()


def _detect_samples(columns: list) -> list:
    """
    Détecte les noms de patients depuis les colonnes beta_<sample>.
    Même logique que compute_stats.py pour être cohérent.
    """
    samples = []
    for c in columns:
        if c.startswith("beta_"):
            sname = c[len("beta_"):]
            if f"cov_{sname}" in columns:
                samples.append(sname)
    if not samples:
        log.error(
            "Aucun patient détecté. Vérifiez que le fichier contient des "
            "colonnes beta_<sample> et cov_<sample>."
        )
        sys.exit(1)
    return samples


def _write_matrix(M: pd.DataFrame, outdir: str, tag: str, fmt: str) -> str:
    """
    Écrit la matrice de clustering patients × sites sur disque.

    Orientation de stockage :
        TSV     → patients × sites (lignes = patients, colonnes = site_ids).
                  Pratique pour inspection manuelle, mais lent sur >>100k sites.
        Parquet → transposée en sites × patients (lignes = site_ids, colonnes
                  = patients + colonne index 'site_id').
                  Raison : Parquet est un format colonnaire. Avec millions de
                  lignes et ~40-100 colonnes, chaque colonne-patient est stockée
                  en un bloc contigu compressé → lecture très rapide même sur
                  de très grandes matrices. L'orientation inverse (patients en
                  lignes, millions de colonnes) produirait un fichier Parquet
                  dégénéré avec autant de métadonnées que de données utiles, et
                  annulerait tout le bénéfice du format colonnaire.
                  Précision float32 : les beta values ont 4 décimales utiles
                  (format "%.4f" dans le TSV legacy) ; float32 offre 7 décimales
                  significatives — sans perte d'information pertinente, mais avec
                  une réduction de 50% de la taille mémoire et disque vs float64.
    """

    if fmt == "parquet":
        out = os.path.join(outdir, f"clustering_matrix{_suffix(tag)}.parquet")
        # Transposer : sites × patients pour l'orientation Parquet optimale.
        # Après M.T, l'index est les site_ids et les colonnes sont les patients.
        # reset_index() fait passer site_id en colonne ordinaire nommée "site_id"
        # (le nom de l'index de M.T, hérité du nom des colonnes de M).
        M_t = M.T.reset_index()
        M_t.columns = M_t.columns.astype(str)
        # La première colonne contient les site_ids (nom variable selon l'index
        # de M) — on la renomme "site_id" pour garantir un nom stable.
        first_col = M_t.columns[0]
        if first_col != "site_id":
            M_t = M_t.rename(columns={first_col: "site_id"})
        float_cols = [c for c in M_t.columns if c != "site_id"]
        pl_df = (
            pl.from_pandas(M_t)
            .with_columns(
                [pl.col(c).cast(pl.Float32) for c in float_cols]
            )
        )
        pl_df.write_parquet(out, compression="zstd", compression_level=3)
    else:
        out = os.path.join(outdir, f"clustering_matrix{_suffix(tag)}.tsv")
        M.to_csv(out, sep="\t", float_format="%.4f")

    return out


def _load_matrix(path: str) -> pd.DataFrame:
    """
    Charge la matrice de clustering en patients × sites (float32).

    Détecte le format (Parquet / TSV) à partir de l'extension.
    Parquet : lu via Polars (scan_parquet → collect), transposé en mémoire.
    TSV     : fallback pd.read_csv pour compatibilité ascendante avec les
              fichiers produits par les versions antérieures du script.

    Les NaN sont autorisés ici : les sous-commandes PCA/UMAP/k-means les
    traitent ensuite en retirant les sites qui en contiennent au moins un.
    """
    if path.endswith(".parquet"):
        # Polars lit le fichier colonnaire en parallèle (multithreaded)
        pl_df   = pl.read_parquet(path)
        site_ids = pl_df["site_id"].to_numpy()
        samples  = [c for c in pl_df.columns if c != "site_id"]
        # Reconstruction : numpy (sites × patients) → transposition → pandas
        data = pl_df.select(samples).to_numpy().T   # patients × sites
        df   = pd.DataFrame(data, index=samples,
                            columns=site_ids, dtype=np.float32)
        df.index.name = "sample"
    else:
        df = pd.read_csv(path, sep="\t", index_col=0)

    log.info("Matrice chargée : %d patients × %d sites", *df.shape)
    n_nan = int(np.isnan(df.values).sum())
    if n_nan > 0:
        log.info(
            "%d NaN détectés dans la matrice ; les sites concernés seront "
            "retirés avant PCA/UMAP/k-means.",
            n_nan,
        )
    return df


def _drop_nan_sites(M: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Retire les sites (colonnes) contenant au moins un NaN.

    Retourne : (matrice_filtrée, n_sites_supprimés, n_sites_initial)
    """
    n_total = int(M.shape[1])
    if n_total == 0:
        return M, 0, 0

    nan_site_mask = M.isna().any(axis=0)
    n_drop = int(nan_site_mask.sum())
    if n_drop > 0:
        M_clean = M.loc[:, ~nan_site_mask].copy()
        log.info(
            "  %d/%d sites supprimés car contenant au moins un NaN (%.2f%%)",
            n_drop, n_total, 100.0 * n_drop / n_total,
        )
    else:
        M_clean = M.copy()
        log.info("  0/%d sites supprimés car contenant au moins un NaN (0.00%%)", n_total)

    return M_clean, n_drop, n_total


def _load_matrix_with_nan(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Charge une matrice produite par prepare --no-impute (NaN autorisés)
    sous forme de tableau NumPy patients × sites.

    Contrairement à la version DataFrame, cette variante évite la création
    d'un DataFrame pandas intermédiaire et la transposition coûteuse sur un
    objet pandas, ce qui réduit fortement le temps de chargement sur les
    grandes matrices parquet.
    """
    if path.endswith(".parquet"):
        pl_df   = pl.read_parquet(path)
        samples = [c for c in pl_df.columns if c != "site_id"]
        # Le fichier parquet est stocké en orientation sites × patients.
        # On le charge directement en tableau NumPy patients × sites.
        values = pl_df.select(samples).to_numpy().T.astype(np.float32, copy=False)
    else:
        df = pd.read_csv(path, sep="\t", index_col=0)
        samples = df.index.tolist()
        values = df.to_numpy(dtype=np.float32, copy=False)

    n_nan   = int(np.isnan(values).sum())
    n_total = values.shape[0] * values.shape[1]
    log.info(
        "Matrice chargée : %d patients × %d sites | %d NaN (%.1f%%)",
        values.shape[0], values.shape[1], n_nan, 100 * n_nan / n_total,
    )
    return values, samples


def _pairwise_euclidean(M: np.ndarray, min_coobs: int) -> np.ndarray:
    """
    Calcul de la distance euclidienne patients × patients.
    """
    if not np.isnan(M).any():
        M = M.astype(np.float64, copy=False)
        norms = np.einsum("ij,ij->i", M, M)
        sq_dist = norms[:, None] + norms[None, :] - 2.0 * (M @ M.T)
        sq_dist = np.maximum(sq_dist, 0.0)
        D = np.sqrt(sq_dist)
        np.fill_diagonal(D, 0.0)
        return D

    n, p = M.shape

    obs  = (~np.isnan(M)).astype(np.float32)
    Mf   = np.where(np.isnan(M), 0.0, M).astype(np.float32)
    n_co = obs @ obs.T
    sq      = Mf ** 2
    sq_sum  = sq @ obs.T
    cross   = Mf @ Mf.T

    sq_dist = sq_sum + sq_sum.T - 2.0 * cross
    sq_dist = np.maximum(sq_dist, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.sqrt(np.where(n_co > 0, p / n_co * sq_dist, np.nan))
    np.fill_diagonal(D, 0.0)
    return D

def _rank_transform_available(M: np.ndarray) -> np.ndarray:
    """
    Transforme chaque ligne (patient) en rangs, en ignorant les NaN.

    Nécessaire pour Spearman available-case : rankdata ne gère pas les NaN
    nativement (il les inclurait dans le rang, faussant toute la ligne).
    Rangs "average" en cas d'ex-aequo — cohérent avec des beta values
    discrétisées par la profondeur de lecture (deux sites peuvent partager
    exactement le même beta chez un patient, ex. 3/10 reads).
    """
    n, p = M.shape
    R = np.full((n, p), np.nan, dtype=np.float64)
    for i in range(n):
        valid = ~np.isnan(M[i])
        if valid.sum() > 0:
            R[i, valid] = rankdata(M[i, valid], method="average")
    return R


def _log_coobs_diagnostic(n_co: np.ndarray, min_coobs: int, n: int, metric_label: str):
    """
    Diagnostic de fiabilité des distances, factorisé pour être réutilisé
    par toutes les métriques (euclidienne et corrélation).
    """
    low_mask = n_co < min_coobs
    np.fill_diagonal(low_mask, False)
    n_low = int(low_mask.sum()) // 2
    n_pairs = n * (n - 1) // 2

    if n_low > 0:
        log.warning(
            "  [%s] %d paires sur %d (%.1f%%) ont moins de %d sites "
            "co-observés. Distance instable pour ces paires ; envisager "
            "d'abaisser --min-completeness ou d'exclure les patients "
            "concernés.",
            metric_label, n_low, n_pairs, 100 * n_low / n_pairs, min_coobs,
        )
    else:
        off_diag = n_co[~np.eye(n, dtype=bool)]
        min_val = int(off_diag.min()) if off_diag.size else 0
        log.info(
            "  [%s] toutes les %d paires ont >= %d sites co-observés (min observé : %d)",
            metric_label, n_pairs, min_coobs, min_val,
        )


def _correlation_pairwise_available(M: np.ndarray, metric_label: str = "pearson") -> tuple:
    """
    Distance 1 - corrélation de Pearson, calculée uniquement sur les sites
    co-observés entre chaque paire de patients (available-case).

    À utiliser directement pour Pearson, ou sur une matrice pré-transformée
    en rangs (_rank_transform_available) pour Spearman.

    Contrairement à la distance euclidienne, le centrage/normalisation
    d'une corrélation est spécifique à l'ensemble des sites co-observés de
    CHAQUE paire (i,j) — aucune factorisation matricielle exacte commune à
    toutes les paires n'existe (contrairement à d²(i,j) = Σβ²_i + Σβ²_j -
    2Σβ_iβ_j qui se décompose en produits matriciels indépendants de la
    paire). Calcul donc effectué paire par paire (n choose 2 paires), mais
    chaque paire est vectorisée sur l'axe des sites via NumPy : avec
    n~78 patients, 3003 paires, coût négligeable même pour p élevé.

    Retourne (D, n_co).
    """
    n, p = M.shape
    obs = ~np.isnan(M)
    D = np.zeros((n, n), dtype=np.float64)
    n_co = np.zeros((n, n), dtype=np.int64)

    n_undefined = 0
    for i in range(n):
        n_co[i, i] = int(obs[i].sum())
        for j in range(i + 1, n):
            co = obs[i] & obs[j]
            n_ij = int(co.sum())
            n_co[i, j] = n_co[j, i] = n_ij

            if n_ij < 2:
                # Corrélation indéfinie avec < 2 points communs.
                n_undefined += 1
                D[i, j] = D[j, i] = np.nan
                continue

            xi = M[i, co]
            xj = M[j, co]
            xi_c = xi - xi.mean()
            xj_c = xj - xj.mean()
            denom = np.sqrt((xi_c ** 2).sum() * (xj_c ** 2).sum())

            if denom < 1e-12:
                r = 0.0
            else:
                r = float((xi_c * xj_c).sum() / denom)
                r = max(-1.0, min(1.0, r))

            D[i, j] = D[j, i] = 1.0 - r

    if n_undefined > 0:
        log.warning(
            "  [%s] %d paires avec < 2 sites co-observés — dissimilarité "
            "non définie (NaN) ; ces paires apparaîtront aussi comme "
            "sous le seuil min_coobs dans le diagnostic.",
            metric_label, n_undefined,
        )

    np.fill_diagonal(D, 0.0)
    return D, n_co


def _load_metadata(path, samples: list):
    if path is None:
        return None
    meta = pd.read_csv(path, sep="\t", index_col="sample")
    missing = set(samples) - set(meta.index)
    if missing:
        log.warning(
            "Métadonnées absentes pour %d patients : %s",
            len(missing), ", ".join(sorted(missing)[:5]),
        )
    return meta.reindex(samples)


def _suffix(tag):
    return f"_{tag}" if tag else ""


def _color_vector(samples, meta, color_by, labels=None):
    """
    Retourne (colors, legend_patches).
    Priorité : metadata color_by > labels k-means > index patient.
    """
    if meta is not None and color_by and color_by in meta.columns:
        log.info("On a bien les metadonnées et la colonne color_by est présente dans meta")
        groups  = meta[color_by].fillna("N/A").tolist()
        uniq    = sorted(set(groups))
        print(uniq)
        cmap    = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(uniq)}
        colors  = [cmap[g] for g in groups]
        patches = [mpatches.Patch(color=cmap[g], label=g) for g in uniq]
        log.info("Coloration par métadonnées : %d groupes détectés dans la colonne '%s'", len(uniq), color_by)
        return colors, patches

    if labels is not None:
        uniq    = sorted(set(labels))
        cmap    = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(uniq)}
        colors  = [cmap[k] for k in labels]
        patches = [mpatches.Patch(color=cmap[k], label=f"Cluster {k}") for k in uniq]
        return colors, patches

    cmap    = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(samples)}
    colors  = [cmap[s] for s in samples]
    patches = [mpatches.Patch(color=cmap[s], label=s) for s in samples]
    return colors, patches


def _scatter_plot(coords, samples, colors, patches, xlabel, ylabel, title, out_path, label_bool,
                  color_mode="categorical", cmap_norm=None, colorbar_label=None):
    """
    Repère orthonormé (set_aspect('equal')) : PC1/PC2 (et UMAP1/UMAP2) sont
    deux axes orthogonaux exprimés dans la même unité — un aspect ratio
    libre déforme visuellement les distances et les angles réels entre
    patients, et fausse la lecture de la dispersion (PC1 portant presque
    toujours plus de variance que PC2, l'effet de déformation est quasi
    systématique sans cette correction). Pour UMAP, l'orthonormalité évite
    une distorsion supplémentaire, mais ne rend pas les distances UMAP
    globalement interprétables (l'algorithme ne préserve que la structure
    de voisinage local).

    Labels : si adjustText est installé, les noms de patients sont répartis
    automatiquement pour minimiser les chevauchements (répulsion itérative),
    avec des flèches reliant le label à son point quand il a été déplacé.
    Sinon, repli sur un simple décalage fixe (comportement historique).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=90,
                         edgecolors="white", linewidths=0.6, zorder=3)

    if color_mode == "continuous" and cmap_norm is not None:
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=cmap_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label(colorbar_label or "Valeur", fontsize=12)
    elif patches:
        ax.legend(handles=patches, bbox_to_anchor=(1.02, 1),
                  loc="upper left", fontsize=11, framealpha=0.85)

    if label_bool == "True":
        if _HAS_ADJUSTTEXT:
            texts = [
                ax.text(coords[i, 0], coords[i, 1], name, fontsize=9, alpha=0.9)
                for i, name in enumerate(samples)
            ]
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.65),
                expand=(1.3, 1.5),
            )
        else:
            for i, name in enumerate(samples):
                ax.annotate(name, (coords[i, 0], coords[i, 1]),
                            fontsize=9, ha="left", va="bottom",
                            xytext=(5, 5), textcoords="offset points", alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)


# ================== IMPUTATION (patients × sites) =========
#
# Les trois fonctions ci-dessous opèrent toutes sur une matrice M orientée
# patients (lignes) × sites (colonnes) — la même orientation que celle
# utilisée en aval par PCA/k-means (patients = observations, sites =
# features). Elles retournent une matrice complète (sans NaN), bornée à
# [0, 1] car beta est une proportion.

def _impute_median(M: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation par médiane du site (sur les patients couverts).

    Conservée pour compatibilité et comparaison, mais déconseillée comme
    méthode principale pour le clustering : chaque site est traité
    indépendamment, ce qui écrase la covariance inter-sites. Combinée à une
    sélection de sites par variance, ce traitement biaise la sélection vers
    les sites où l'imputation introduit le plus de bruit relatif plutôt que
    vers les sites biologiquement les plus informatifs — c'est très
    probablement la cause du scree plot plat (PC1 ~3.5%) et des clusters
    singletons observés sur la cohorte à 16 patients.
    """
    site_medians = M.median(axis=0, skipna=True)
    return M.fillna(site_medians)


def _impute_knn(M: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    """
    Imputation par k plus proches voisins entre patients.

    Pour chaque valeur manquante (patient p, site s), la distance euclidienne
    entre p et chaque autre patient est calculée sur les sites co-observés
    uniquement (cf. Troyanskaya et al. 2001, méthode standard pour les
    matrices d'expression à trous), puis la valeur manquante est imputée par
    la moyenne pondérée par distance des n_neighbors patients les plus
    proches au même site.

    Contrairement à la médiane par site, cette méthode exploite la
    similarité du profil épigénétique global du patient plutôt qu'une
    statistique marginale du site qui ignore toute structure inter-patients.
    Pertinent quand on suppose que les patients partagent une structure de
    sous-groupes (l'hypothèse même du clustering) : un patient est imputé en
    se rapprochant des patients dont il est globalement le plus proche.

    Limite : suppose implicitement que la similarité globale (calculée sur
    tous les sites retenus) est informative même avant d'avoir identifié les
    sous-groupes — risque de lissage circulaire si n_neighbors est trop
    petit par rapport à la taille de cohorte. Pour n petit (<20 patients),
    préférer --imputation-method pca.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    X = imputer.fit_transform(M.to_numpy(dtype=float))
    X = np.clip(X, 0.0, 1.0)
    return pd.DataFrame(X, index=M.index, columns=M.columns)


def _impute_pca_em(
    M: pd.DataFrame,
    n_components: int,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> pd.DataFrame:
    """
    Imputation par PCA itérative (algorithme EM / iterativeSVD), équivalent
    à missMDA::imputePCA en R (Josse & Husson, 2012, J. SFdS).

    Principe : contrairement à l'imputation par médiane, qui traite chaque
    site indépendamment, cette méthode reconstruit les valeurs manquantes en
    exploitant la structure de corrélation entre patients capturée par les
    n_components premières composantes principales. Elle préserve donc la
    covariance inter-sites au lieu de l'écraser, ce qui la rend cohérente
    avec l'usage en aval (la matrice imputée sert justement à une PCA).

    Algorithme :
      1. Initialisation : NaN remplacés par la moyenne du site (sur patients
         couverts).
      2. PCA à n_components sur la matrice courante.
      3. Reconstruction X_hat = scores @ loadings + moyenne.
      4. Seules les positions originellement manquantes sont remplacées par
         X_hat ; les valeurs observées restent inchangées à chaque itération.
      5. Répéter 2-4 jusqu'à convergence (variation relative de la norme de
         Frobenius < tol) ou max_iter atteint.

    C'est la méthode recommandée par défaut : elle est cohérente avec
    l'observation que pour une cohorte de taille modérée (~40 patients), la
    matrice complète (sans présélection de sites par variance) est
    biologiquement préférable, et elle évite le biais d'écrasement de
    covariance introduit par l'imputation médiane.
    
    Optimisation NumPy : initialisation et itérations effectuées directement
    sur les arrays NumPy (pas d'allers-retours DataFrame → NumPy) pour une
    vectorisation complète et une performance ~2-3× supérieure sur les grandes
    matrices (>100k sites).
    """
    mask = M.isna()
    n_missing = int(mask.to_numpy().sum())
    if n_missing == 0:
        return M

    # Conversion une seule fois en NumPy ; initialisation par np.nanmean (vectorisé)
    # au lieu de M.mean().fillna() (qui itère colonne-par-colonne chez pandas).
    X = M.to_numpy(dtype=np.float64, copy=True)
    col_means = np.nanmean(X, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = col_means[nan_cols]
    mask_arr = mask.to_numpy()

    n_comp = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    if n_comp < n_components:
        log.warning(
            "  PCA itérative : n_components réduit à %d (limité par "
            "n_patients-1=%d ou n_sites=%d)",
            n_comp, X.shape[0] - 1, X.shape[1],
        )

    delta = np.inf
    for it in range(1, max_iter + 1):
        mu = X.mean(axis=0)
        Xc = X - mu
        pca = PCA(n_components=n_comp, random_state=42)
        scores = pca.fit_transform(Xc)
        X_hat = scores @ pca.components_ + mu

        # Ne remplacer que les positions originellement manquantes :
        # les valeurs réellement observées ne doivent jamais être altérées.
        X_new = np.where(mask_arr, X_hat, X)

        denom = np.linalg.norm(X) + 1e-12
        delta = np.linalg.norm(X_new - X) / denom
        X = X_new

        if delta < tol:
            log.info(
                "  PCA itérative : convergence à l'itération %d (delta=%.2e)",
                it, delta,
            )
            break
    else:
        log.warning(
            "  PCA itérative : max_iter=%d atteint sans convergence (delta=%.2e). "
            "Augmenter --pca-max-iter ou vérifier --pca-n-components.",
            max_iter, delta,
        )

    # Beta est une proportion bornée [0, 1] ; la reconstruction PCA peut
    # légèrement dépasser ces bornes — on les impose biologiquement.
    n_clipped = int(((X < 0) | (X > 1)).sum())
    if n_clipped > 0:
        log.info(
            "  PCA itérative : %d valeurs reconstruites hors [0,1] ramenées "
            "aux bornes biologiques",
            n_clipped,
        )
    X = np.clip(X, 0.0, 1.0)

    return pd.DataFrame(X, index=M.index, columns=M.columns)

def _impute_random(M: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation aléatoire uniforme sur [0, 1] pour diagnostic de robustesse.
    """
    mask = M.isna()
    n_missing = int(mask.to_numpy().sum())
    if n_missing == 0:
        return M

    X = np.array(M.to_numpy(dtype=float), copy=True)
    mask_arr = mask.to_numpy()
    random_values = np.random.uniform(0.0, 1.0, size=X.shape)
    X[mask_arr] = random_values[mask_arr]
    return pd.DataFrame(X, index=M.index, columns=M.columns)


def _site_completeness(M: pd.DataFrame) -> pd.Series:
    """
    Fraction de patients non-NaN, par site (colonne).
    Calculée une seule fois et réutilisée pour le diagnostic et le filtrage,
    afin que le diagnostic reflète exactement ce qui sera filtré.
    """
    return M.notna().sum(axis=0) / M.shape[0]


def _log_completeness_distribution(frac_observed: pd.Series):
    """
    Logue les quantiles de la distribution de complétude par site, avant
    tout filtrage. Indispensable avant de choisir min_completeness :
    avec une grande cohorte (N patients élevé), le nombre de patients
    couverts par site suit approximativement une loi binomiale qui se
    concentre fortement autour de sa moyenne (écart-type ~ sqrt(N*q*(1-q))).
    Un effet de seuil très abrupt est donc normal — un déplacement de
    quelques points du seuil peut faire passer le nombre de sites retenus
    de la quasi-totalité à presque rien, sans zone de transition douce.
    Inspecter ces quantiles avant de fixer le seuil évite de choisir une
    valeur arbitraire qui tombe dans la zone de chute brutale.
    """
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    quantiles = frac_observed.quantile(qs)
    summary = ", ".join(f"p{int(q*100)}={v:.3f}" for q, v in quantiles.items())
    log.info("  distribution de complétude par site (avant filtre) : %s", summary)


def _write_completeness_histogram(frac_observed: pd.Series, path: str, n_bins: int = 20):
    """
    Écrit un histogramme (comptage par tranche de complétude) plutôt que la
    liste complète par site, pour rester léger même avec des millions de
    sites. À tracer (barplot bin_low/bin_high vs n_sites) pour repérer
    visuellement si la distribution est bimodale (ex. adaptive sampling :
    un pic hors-cible proche de 0 et un pic sur-cible proche de 1) — dans ce
    cas, le seuil de complétude doit être placé dans la vallée entre les
    deux modes, pas à une valeur ronde arbitraire comme 0.5 ou 0.7.
    """
    counts, edges = np.histogram(frac_observed.to_numpy(), bins=n_bins, range=(0.0, 1.0))
    hist_df = pd.DataFrame({
        "bin_low":  edges[:-1],
        "bin_high": edges[1:],
        "n_sites":  counts,
    })
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    hist_df.to_csv(path, sep="\t", index=False)
    log.info("  histogramme de complétude (%d tranches) écrit → %s", n_bins, path)


def _completeness_filter(M: pd.DataFrame, frac_observed: pd.Series, min_completeness: float) -> pd.DataFrame:
    """
    Filtre les sites (colonnes) dont la fraction de patients non-NaN est
    inférieure à min_completeness.

    Justification statistique : ce filtre est distinct du filtre semi-strict
    déjà appliqué en amont (au niveau du pipeline methylation_cohort.tsv,
    qui garantit ≥ ceil(min_cov_frac × N) patients couverts par site). Ici,
    l'objectif est différent : limiter spécifiquement la part de valeurs
    imputées dans la matrice de clustering, quelle que soit la méthode
    d'imputation choisie. Plus un site a de NaN, plus son imputation est une
    extrapolation plutôt qu'une estimation — un site imputé à 90% n'apporte
    quasiment aucune information patient-spécifique réelle, qu'on le sache
    ou non au moment de l'interpréter.

    ATTENTION : avec un grand nombre de patients, ce filtre a un effet de
    seuil très abrupt (cf. _log_completeness_distribution) — toujours
    inspecter la distribution avant de fixer min_completeness, plutôt que
    d'utiliser la valeur par défaut sans vérification sur des cohortes de
    grande taille ou des données à couverture très hétérogène (ex. adaptive
    sampling).
    """
    keep = frac_observed >= min_completeness
    n_dropped = int((~keep).sum())
    log.info(
        "  filtre de complétude (>= %.0f%% patients non-NaN) : "
        "%d sites conservés, %d retirés",
        min_completeness * 100, int(keep.sum()), n_dropped,
    )
    return M.loc[:, keep]


# ========================= STEP 1 : prepare ==============

def cmd_prepare(args):
    """
    Lit methylation_cohort.tsv, filtre sur mod_code, filtre les sites par
    complétude, impute les NaN résiduels, sélectionne optionnellement les
    top N sites les plus variables, et produit une matrice patients × sites.

    Pas d'agrégation des brins : le pipeline a déjà conservé un seul brin
    par position (le plus couvert) dans parse_sample.

    Imputation : avec le filtre semi-strict amont, beta=NaN signifie que le
    patient avait N_valid < min_cov sur ce site. --imputation-method
    contrôle la méthode utilisée pour ces NaN résiduels (median / knn / pca,
    cf. docstrings des fonctions _impute_*).
    """
    os.makedirs(args.outdir, exist_ok=True)
    log.info("prepare : chargement de %s…", args.cohort_tsv)
    df = pd.read_csv(args.cohort_tsv, sep="\t", low_memory=False)
    log.info("  %d lignes, %d colonnes", *df.shape)

    # ── Détection des patients ───────────────────────────
    samples = _detect_samples(list(df.columns))
    log.info("  %d patients détectés : %s", len(samples), samples)

    if args.exclude_samples:
        exclude_set = set(args.exclude_samples)
        known_excluded = [s for s in samples if s in exclude_set]
        unknown_excluded = [s for s in args.exclude_samples if s not in exclude_set]
        if known_excluded:
            samples = [s for s in samples if s not in exclude_set]
            log.info(
                "  exclusion de %d patients : %s",
                len(known_excluded), ", ".join(known_excluded)
            )
        if unknown_excluded:
            log.warning(
                "  patients inconnus ignorés : %s",
                ", ".join(unknown_excluded),
            )
        if not samples:
            log.error("Tous les patients sont exclus. Veuillez vérifier --exclude-samples.")
            sys.exit(1)

    beta_cols = [f"beta_{s}" for s in samples]

    # ── Filtre mod_code ──────────────────────────────────
    if "mod_code" not in df.columns:
        log.error("Colonne 'mod_code' absente du TSV. Arrêt.")
        sys.exit(1)

    n_before = len(df)
    df = df[df["mod_code"] == args.mod_code].copy()
    log.info(
        "  après filtre mod_code='%s' : %d sites (supprimé %d)",
        args.mod_code, len(df), n_before - len(df),
    )

    if df.empty:
        log.error(
            "Aucun site pour mod_code='%s'. "
            "Vérifiez que ce mod_code est présent dans le TSV.",
            args.mod_code,
        )
        sys.exit(1)

    # ── Conversion numérique ("NaN" chaîne → np.nan) ────
    beta_mat = df[beta_cols].apply(pd.to_numeric, errors="coerce")

    fully_nan_sites = beta_mat.isna().all(axis=1)
    if fully_nan_sites.any():
        n_fully_nan_sites = int(fully_nan_sites.sum())
        df = df.loc[~fully_nan_sites].copy()
        beta_mat = beta_mat.loc[~fully_nan_sites]
        log.info(
            "  %d sites supprimés car couverts uniquement par des patients exclus",
            n_fully_nan_sites,
        )
        if df.empty:
            log.error("Aucun site restant après exclusion des patients. Arrêt.")
            sys.exit(1)

    # ── Passage en orientation patients × sites ──────────
    # Toutes les étapes suivantes (filtre de complétude, imputation,
    # sélection de variance) opèrent dans cette orientation : cohérente avec
    # l'espace utilisé en aval par PCA/k-means (patients = observations,
    # sites = features), et nécessaire pour que kNN/PCA-EM exploitent la
    # similarité entre patients plutôt qu'une statistique par site isolé.
    M = beta_mat.T
    M.columns = df["site_id"].values
    M.index = samples
    M.index.name = "sample"

    n_nan_total = int(M.isna().sum().sum())
    n_total_cells = M.shape[0] * M.shape[1]
    log.info(
        "  %d valeurs NaN sur %d sites x %d patients (%.1f%%) avant filtre de complétude",
        n_nan_total, M.shape[1], M.shape[0],
        100 * n_nan_total / n_total_cells if n_total_cells else 0.0,
    )

    # ── Diagnostic + filtre de complétude ─────────────────
    frac_observed = _site_completeness(M)
    _log_completeness_distribution(frac_observed)
    if args.completeness_histogram:
        _write_completeness_histogram(frac_observed, args.completeness_histogram, n_bins=20)

    M = _completeness_filter(M, frac_observed, args.min_completeness)
    if M.shape[1] == 0:
        log.error(
            "Aucun site ne satisfait le seuil de complétude %.2f. Arrêt.",
            args.min_completeness,
        )
        sys.exit(1)

    # Garde-fou : le filtre semi-strict amont garantit qu'aucun site retenu
    # n'est entièrement NaN ; on le revérifie ici par sécurité, car le
    # filtre de complétude pourrait en théorie laisser passer un site
    # entièrement vide si min_completeness=0.
    fully_nan_after = M.isna().all(axis=0)
    if fully_nan_after.any():
        log.error(
            "  %d sites sans aucun patient couvert après filtre de complétude — "
            "données incohérentes avec le filtre semi-strict amont. Arrêt.",
            int(fully_nan_after.sum()),
        )
        sys.exit(1)

    # ── Imputation des NaN résiduels (ou conservation intentionnelle) ──────
    n_nan_remaining = int(M.isna().sum().sum())

    if n_nan_remaining > 0:
        n_sites_with_nan = int(M.isna().any(axis=0).sum())
        log.info(
            "  %d NaN résiduels sur %d sites (sur %d retenus) — "
            "imputation par méthode '%s'",
            n_nan_remaining, n_sites_with_nan, M.shape[1], args.imputation_method,
        )

        if args.imputation_method == "no_impute":
            # Mode available-case : on conserve les NaN intentionnellement.
            log.info(
                "  --no-impute : %d NaN conservés dans la matrice de sortie "
                "(%.1f%% de la matrice). Compatibilité : run_available_case uniquement.",
                n_nan_remaining,
                100 * n_nan_remaining / (M.shape[0] * M.shape[1]),
            )

        if args.imputation_method == "median":
            M = _impute_median(M)

        elif args.imputation_method == "knn":
            k = min(args.knn_neighbors, M.shape[0] - 1)
            if k < args.knn_neighbors:
                log.warning(
                    "  knn_neighbors réduit à %d (n_patients-1=%d)",
                    k, M.shape[0] - 1,
                )
            if k < 1:
                log.error("Trop peu de patients pour l'imputation kNN. Arrêt.")
                sys.exit(1)
            M = _impute_knn(M, n_neighbors=k)

        elif args.imputation_method == "pca":
            M = _impute_pca_em(
                M,
                n_components=args.pca_n_components,
                max_iter=args.pca_max_iter,
                tol=args.pca_tol,
            )
        
        elif args.imputation_method == "random":
            M = _impute_random(M)

        n_nan_after = int(M.isna().sum().sum())
        if args.imputation_method != "no_impute":
            assert n_nan_after == 0, f"Imputation incomplète : {n_nan_after} NaN résiduels"
        log.info("  imputation terminée — matrice complète")
    else:
        log.info("  aucun NaN résiduel après filtre de complétude")

    # ── Sélection optionnelle des sites les plus variables ───
    if args.top_n is not None:
        var = M.var(axis=0, ddof=1)
        top_n = min(args.top_n, M.shape[1])
        top_cols = var.nlargest(top_n).index
        M = M[top_cols]
        log.info(
            "  top %d sites sélectionnés par variance "
            "  (médiane : %.4f, min : %.4f, max : %.4f)",
            top_n, var.nlargest(top_n).median(), var.nlargest(top_n).min(), var.nlargest(top_n).max()
        )

    # Diagnostic à faire avant de binariser :
    flat = M.values.flatten()
    flat = flat[~np.isnan(flat)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flat, bins=100)
    ax.set_xlabel("Beta value")
    ax.set_ylabel("Nombre de sites")
    ax.set_title("Distribution globale des beta — décision de binarisation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "distribution_beta.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → distribution_beta.png")

    # ── Binarisation ──────────────────────────
    if args.seuil =="True":
        log.info("Binarisation de la matrice : beta >= 0.5 → 1, beta < 0.5 → 0")
        M = (M >= 0.5).astype(np.float32)

    log.info("  matrice finale : %d patients × %d sites", *M.shape)

    out = _write_matrix(M, args.outdir, args.tag, args.output_format)
    log.info("prepare → %s", out)


# ========================= STEP 2 : run_pca ==============
def _scatter_grid_pca(coords, evr, samples, colors, patches,
                      out_path, label_bool, color_mode="categorical",
                      colorbar_label=None, cmap_norm=None, color_by=None):
    """
    Grille 2×2 (PC1/2, PC1/3, PC2/3, scree) — ou grille 1×3 sans scree
    plot si color_by == "risk" ou == "age"

    Le scree plot n'a de sens que pour guider le choix du nombre de
    composantes à retenir lors d'une exploration structurale
    (clustering, groupes cliniques). Quand la coloration encode un
    score de risque continu, l'objectif de la figure change : on
    cherche à visualiser un gradient dans l'espace PCA, pas à motiver
    un choix de dimensionnalité. Le scree plot devient alors non
    pertinent et est retiré — décision pilotée directement par
    l'argument --color-by transmis par l'utilisateur, indépendamment
    de la présence effective de valeurs de risque valides dans les
    métadonnées (pour rester cohérent même si toutes les valeurs de
    risk sont NaN).

    Layout (show_scree=True) :
        ┌──────────────┬──────────────┐
        │  PC1 vs PC2  │  PC1 vs PC3  │
        ├──────────────┼──────────────┤
        │  PC2 vs PC3  │  Scree plot  │
        └──────────────┴──────────────┘

    Layout (show_scree=False) :
        ┌──────────────┬──────────────┬──────────────┐
        │  PC1 vs PC2  │  PC1 vs PC3  │  PC2 vs PC3  │
        └──────────────┴──────────────┴──────────────┘

    Chaque scatter est en repère orthonormé (set_aspect='equal') :
    PC1, PC2, PC3 sont orthogonaux par construction et exprimés dans
    la même unité (variance standardisée) — un aspect ratio libre
    déformerait les distances et angles réels entre patients.
    """
    n_comp_avail = coords.shape[1]
    pairs = [(0, 1), (0, 2), (1, 2)]   # indices des paires PC à afficher
    show_scree = (color_by != "risk" and color_by != "age")

    if show_scree:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]

        if i >= n_comp_avail or j >= n_comp_avail:
            ax.set_visible(False)
            continue

        xi, xj = coords[:, i], coords[:, j]
        ax.scatter(xi, xj, c=colors, s=70,
                   edgecolors="white", linewidths=0.4, zorder=3)

        if label_bool == "True":
            if _HAS_ADJUSTTEXT:
                texts = [
                    ax.text(xi[k], xj[k], name, fontsize=8, alpha=0.85)
                    for k, name in enumerate(samples)
                ]
                adjust_text(
                    texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="grey",
                                    lw=0.5, alpha=0.55),
                    expand=(1.2, 1.4),
                )
            else:
                for k, name in enumerate(samples):
                    ax.annotate(name, (xi[k], xj[k]),
                                fontsize=8, ha="left", va="bottom",
                                xytext=(4, 4), textcoords="offset points",
                                alpha=0.8)

        ax.set_xlabel(f"PC{i+1} ({evr[i]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC{j+1} ({evr[j]*100:.1f}%)", fontsize=12)
        ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(labelsize=11)

    right_margin = 0.80
    if color_mode == "continuous" and cmap_norm is not None:
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=cmap_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:3] if not show_scree else axes,
                            shrink=0.85, pad=0.02,
                            fraction=0.05, aspect=25)
        cbar.set_label(colorbar_label or "Valeur", fontsize=12)
    elif patches:
        fig.legend(handles=patches, fontsize=12, framealpha=0.88,
                   loc="center left", bbox_to_anchor=(right_margin + 0.01, 0.5),
                   bbox_transform=fig.transFigure,
                   title="Groupes", title_fontsize=12)

    if show_scree:
        # Panneau 4 : scree plot
        ax_scree = axes[3]
        n_show   = min(len(evr), 15)       # max 15 composantes affichées
        x_pos    = np.arange(1, n_show + 1)
        var_pct  = evr[:n_show] * 100
        cum_pct  = np.cumsum(evr[:n_show]) * 100

        bars = ax_scree.bar(x_pos, var_pct, color="#3B8BD4", alpha=0.8,
                            label="Variance expliquée")
        ax2  = ax_scree.twinx()
        ax2.plot(x_pos, cum_pct, "o-", color="#E8593C",
                 linewidth=2, markersize=6, label="Variance cumulée")

        for b, v in zip(bars[:3], var_pct[:3]):
            ax_scree.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                          f"{v:.1f}%", ha="center", va="bottom",
                          fontsize=9, color="#3B8BD4", fontweight="bold")

        ax_scree.set_xlabel("Composante principale", fontsize=12)
        ax_scree.set_ylabel("Variance expliquée (%)", color="#3B8BD4", fontsize=11)
        ax2.set_ylabel("Variance cumulée (%)", color="#E8593C", fontsize=11)
        ax_scree.set_xticks(x_pos)
        ax_scree.tick_params(labelsize=11)
        ax_scree.set_title("Scree plot", fontsize=13, fontweight="bold")
        ax_scree.grid(True, linestyle="--", alpha=0.3, axis="y")

        for pct, ls in [(50, "--"), (80, ":")]:
            ax2.axhline(pct, color="#E8593C", linestyle=ls,
                        linewidth=0.8, alpha=0.5)
            ax2.text(n_show, pct + 1, f"{pct}%",
                     color="#E8593C", fontsize=7, ha="right", va="bottom")

    fig.suptitle("PCA — profils de méthylation", fontsize=14,
                 fontweight="bold", y=1.01)

    fig.tight_layout(rect=(0, 0, right_margin, 1))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out_path)

def cmd_run_pca(args):
    log.info("run_pca : chargement de la matrice…")
    matrix  = _load_matrix(args.matrix)
    samples = matrix.index.tolist()
    meta    = _load_metadata(args.metadata, samples)

    matrix, n_drop, n_total = _drop_nan_sites(matrix)
    if matrix.shape[1] == 0:
        log.error("Aucun site conservé après suppression des NaN. Arrêt.")
        sys.exit(1)

    X      = _get_scaler(args.use_robust_scaler).fit_transform(matrix.values)
    n_comp = min(args.n_components, X.shape[0], X.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X)

    log.info(
        "PCA : variance expliquée PC1=%.1f%% PC2=%.1f%% (cumul 3 comp=%.1f%%)",
        pca.explained_variance_ratio_[0] * 100,
        pca.explained_variance_ratio_[1] * 100,
        pca.explained_variance_ratio_[:min(3, n_comp)].sum() * 100,
    )

    os.makedirs(args.outdir, exist_ok=True)

    col_names = [f"PC{i+1}" for i in range(n_comp)]
    df_coords = pd.DataFrame(coords, index=samples, columns=col_names)
    df_coords.index.name = "sample"
    df_coords.to_csv(os.path.join(args.outdir, f"pca_coords{_suffix(args.tag)}.tsv"),
                     sep="\t", float_format="%.6f")

    df_var = pd.DataFrame({
        "component":           col_names,
        "explained_variance":  pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    df_var.to_csv(os.path.join(args.outdir, f"pca_variance{_suffix(args.tag)}.tsv"),
                  sep="\t", index=False, float_format="%.6f")

    label_bool = True if hasattr(args, "label_bool") and args.label_bool == "True" else False

    if meta is not None and (args.color_by == "risk" and "risk" in meta.columns or args.color_by == "age" and "age" in meta.columns):
        print(f"Coloration continue par {args.color_by} pour la PCA (colonne '{args.color_by}')")
        if args.color_by == "risk":
            vals = pd.to_numeric(meta["risk"], errors="coerce")
        else:
            vals = pd.to_numeric(meta["age"], errors="coerce")
        valid_mask = vals.notna()
        if valid_mask.any():
            vmin = float(vals[valid_mask].min())
            vmax = float(vals[valid_mask].max())
            cmap_norm = Normalize(vmin=vmin, vmax=vmax)
            colors = [plt.cm.viridis(cmap_norm(v)) if pd.notna(v) else "#cccccc"
                      for v in vals]
            patches = None
            color_mode = "continuous"
            colorbar_label = "Score de risque de Manchester"
            log.info("Coloration continue par %s pour la PCA (colonne '%s')", args.color_by, args.color_by)
        else:
            colors, patches = _color_vector(samples, meta, args.color_by)
            color_mode = "categorical"
            colorbar_label = None
            cmap_norm = None
    else:
        colors, patches = _color_vector(samples, meta, args.color_by)
        color_mode = "categorical"
        colorbar_label = None
        cmap_norm = None
    print(f"Est ce qu'il y a les labels sur les figures ? {label_bool}")
 
    # Figure principale : grille 2×2 (PC1/2, PC1/3, PC2/3, scree)
    _scatter_grid_pca(
        coords=coords,
        evr=pca.explained_variance_ratio_,
        samples=samples,
        colors=colors,
        patches=patches,
        out_path=os.path.join(args.outdir, f"pca_scatter{_suffix(args.tag)}.png"),
        label_bool=label_bool,
        color_mode=color_mode,
        colorbar_label=colorbar_label,
        cmap_norm=cmap_norm,
        color_by=args.color_by,
    )


# ========================= STEP 3 : run_umap =============

def cmd_run_umap(args):
    log.info("run_umap : chargement de la matrice…")
    matrix  = _load_matrix(args.matrix)
    samples = matrix.index.tolist()
    meta    = _load_metadata(args.metadata, samples)

    matrix, n_drop, n_total = _drop_nan_sites(matrix)
    if matrix.shape[1] == 0:
        log.error("Aucun site conservé après suppression des NaN. Arrêt.")
        sys.exit(1)

    X     = _get_scaler(args.use_robust_scaler).fit_transform(matrix.values)
    n_pca = min(50, X.shape[0] - 1, X.shape[1])
    log.info("  PCA préalable à %d composantes avant UMAP…", n_pca)
    X_pca = PCA(n_components=n_pca, random_state=args.random_state).fit_transform(X)

    log.info("  UMAP (n_neighbors=%d, min_dist=%.2f)…",
             args.n_neighbors, args.min_dist)
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=2,
        random_state=args.random_state,
        verbose=False,
    )
    coords = reducer.fit_transform(X_pca)

    os.makedirs(args.outdir, exist_ok=True)

    df_coords = pd.DataFrame(coords, index=samples, columns=["UMAP1", "UMAP2"])
    df_coords.index.name = "sample"
    df_coords.to_csv(os.path.join(args.outdir, f"umap_coords{_suffix(args.tag)}.tsv"),
                     sep="\t", float_format="%.6f")

    label_bool = True if hasattr(args, "label_bool") and args.label_bool == "True" else False

    if meta is not None and (args.color_by == "risk" and "risk" in meta.columns or
                             args.color_by == "age" and "age" in meta.columns):
        if args.color_by == "risk":
            vals = pd.to_numeric(meta["risk"], errors="coerce")
            colorbar_label = "Score de risque de Manchester"
        else:
            vals = pd.to_numeric(meta["age"], errors="coerce")
            colorbar_label = "Age"

        valid_mask = vals.notna()
        if valid_mask.any():
            vmin = float(vals[valid_mask].min())
            vmax = float(vals[valid_mask].max())
            cmap_norm = Normalize(vmin=vmin, vmax=vmax)
            colors = [plt.cm.viridis(cmap_norm(v)) if pd.notna(v) else "#cccccc"
                      for v in vals]
            patches = None
            color_mode = "continuous"
        else:
            colors, patches = _color_vector(samples, meta, args.color_by)
            color_mode = "categorical"
            colorbar_label = None
            cmap_norm = None
    else:
        colors, patches = _color_vector(samples, meta, args.color_by)
        color_mode = "categorical"
        colorbar_label = None
        cmap_norm = None

    _scatter_plot(
        coords=coords,
        samples=samples,
        colors=colors,
        patches=patches,
        xlabel="UMAP1",
        ylabel="UMAP2",
        title=f"UMAP — profils de méthylation (n_neighbors={args.n_neighbors})",
        out_path=os.path.join(args.outdir, f"umap_scatter{_suffix(args.tag)}.png"),
        label_bool=label_bool,
        color_mode=color_mode,
        cmap_norm=cmap_norm,
        colorbar_label=colorbar_label,
    )


# ========================= STEP 4 : run_kmeans ===========

def _elbow_k(inertias: list, k_range: range) -> int:
    """
    Détection automatique du coude par distance maximale à la droite
    reliant le premier et le dernier point (méthode géométrique).
    """
    x1, y1 = k_range[0],  inertias[0]
    x2, y2 = k_range[-1], inertias[-1]
    dx, dy = x2 - x1, y2 - y1
    norm   = np.sqrt(dx**2 + dy**2)
    distances = [
        abs(dy * k - dx * inertia + x2 * y1 - y2 * x1) / norm
        for k, inertia in zip(k_range, inertias)
    ]
    return k_range[int(np.argmax(distances))]


def _write_cluster_membership(df_labels: pd.DataFrame, outdir: str, tag: str) -> str:
    """
    Écrit un fichier "un cluster par ligne" regroupant les noms de patients,
    en complément de kmeans_labels.tsv (une ligne par patient). Pensé pour
    être lu directement plutôt que de déchiffrer des labels superposés sur
    un scatter plot — surtout utile dès que la cohorte dépasse une dizaine
    de patients ou que plusieurs noms tombent au même endroit visuellement.
    """
    grouped = (
        df_labels.groupby("cluster")["sample"]
        .apply(lambda s: ", ".join(sorted(s)))
        .reset_index()
    )
    grouped["n_patients"] = df_labels.groupby("cluster")["sample"].size().values
    grouped = grouped[["cluster", "n_patients", "sample"]].rename(
        columns={"sample": "patients"}
    )
    out_path = os.path.join(outdir, f"kmeans_cluster_members{_suffix(tag)}.tsv")
    grouped.to_csv(out_path, sep="\t", index=False)
    return out_path


def cmd_run_kmeans(args):
    print(f"Est ce qu'il y a les labels sur les figures ? {args.label_bool}")
    log.info("run_kmeans : chargement de la matrice…")
    matrix  = _load_matrix(args.matrix)
    samples = matrix.index.tolist()
    n       = len(samples)

    matrix, n_drop, n_total = _drop_nan_sites(matrix)
    if matrix.shape[1] == 0:
        log.error("Aucun site conservé après suppression des NaN. Arrêt.")
        sys.exit(1)

    X = _get_scaler(args.use_robust_scaler).fit_transform(matrix.values)

    k_max   = min(args.k_max, n - 1)
    k_min   = min(args.k_min, k_max - 1)
    k_range = range(k_min, k_max + 1)

    log.info("  calcul inertie + silhouette pour k=%d à k=%d…", k_min, k_max)
    inertias, silhouettes = [], []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=args.random_state,
                        n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if k > 1 else np.nan
        silhouettes.append(sil)
        log.info("  k=%d : inertie=%.1f  silhouette=%.4f", k, km.inertia_, sil)

    k_final = args.k_final if args.k_final else _elbow_k(inertias, k_range)
    log.info("  K final retenu : %d", k_final)

    km_final = KMeans(n_clusters=k_final, random_state=args.random_state,
                      n_init=20, max_iter=500)
    labels   = km_final.fit_predict(X)

    os.makedirs(args.outdir, exist_ok=True)

    df_labels = pd.DataFrame({"sample": samples, "cluster": labels})
    df_labels.to_csv(os.path.join(args.outdir, f"kmeans_labels{_suffix(args.tag)}.tsv"),
                     sep="\t", index=False)

    members_path = _write_cluster_membership(df_labels, args.outdir, args.tag)
    log.info("  → %s", members_path)

    df_inertia = pd.DataFrame({
        "k":          list(k_range),
        "inertia":    inertias,
        "silhouette": silhouettes,
    })
    df_inertia.to_csv(os.path.join(args.outdir, f"kmeans_inertia{_suffix(args.tag)}.tsv"),
                      sep="\t", index=False, float_format="%.6f")

    # Elbow + silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, "o-", color="#3B8BD4", linewidth=2)
    ax1.axvline(k_final, color="#E8593C", linestyle="--", alpha=0.7,
                label=f"K retenu = {k_final}")
    ax1.set_xlabel("Nombre de clusters K")
    ax1.set_ylabel("Inertie")
    ax1.set_title("Elbow plot", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.plot(list(k_range), silhouettes, "o-", color="#1D9E75", linewidth=2)
    ax2.axvline(k_final, color="#E8593C", linestyle="--", alpha=0.7,
                label=f"K retenu = {k_final}")
    ax2.set_xlabel("Nombre de clusters K")
    ax2.set_ylabel("Score silhouette")
    ax2.set_title("Silhouette score", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Sélection du K optimal — k-means méthylation",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"kmeans_elbow{_suffix(args.tag)}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → kmeans_elbow%s.png", _suffix(args.tag))

    colors, patches = _color_vector(samples, None, None, labels)

    if args.pca_coords:
        pca_df = pd.read_csv(args.pca_coords, sep="\t", index_col=0).reindex(samples)
        _scatter_plot(
            coords=pca_df[["PC1", "PC2"]].values,
            samples=samples, colors=colors, patches=patches,
            xlabel="PC1", ylabel="PC2",
            title=f"PCA coloré par cluster k-means (K={k_final})",
            out_path=os.path.join(args.outdir, f"pca_scatter_kmeans{_suffix(args.tag)}.png"),
            label_bool=args.label_bool,
        )

    if args.umap_coords:
        umap_df = pd.read_csv(args.umap_coords, sep="\t", index_col=0).reindex(samples)
        _scatter_plot(
            coords=umap_df[["UMAP1", "UMAP2"]].values,
            samples=samples, colors=colors, patches=patches,
            xlabel="UMAP1", ylabel="UMAP2",
            title=f"UMAP coloré par cluster k-means (K={k_final})",
            out_path=os.path.join(args.outdir, f"umap_scatter_kmeans{_suffix(args.tag)}.png"),
            label_bool=args.label_bool,
        )

    log.info(
        "run_kmeans : K=%d | distribution clusters :\n%s",
        k_final,
        df_labels["cluster"].value_counts().sort_index().to_string(),
    )

# ========================= STEP 5 : run_gmm =============
def cmd_run_gmm(args):
    """
    Fit d'un Gaussian Mixture Model (GMM) sur la matrice patients × sites.
    Contrairement à k-means, GMM ne contraint pas les clusters à être
    sphériques (selon covariance_type) et fournit une affectation
    probabiliste (soft clustering) ; seule l'affectation MAP (hard label)
    est conservée ici pour rester comparable aux autres sous-commandes.

    AVERTISSEMENT DIMENSIONNALITÉ : le nombre de features p (sites retenus
    après filtre NaN) est presque toujours très supérieur au nombre de
    patients n. Pour covariance_type='full'/'tied', la matrice de
    covariance par composante est alors structurellement singulière
    (rang ≤ n_k - 1 pour n_k patients dans le cluster k, alors qu'une
    matrice p×p pleine de rang plein nécessite n_k ≥ p+1) : reg_covar
    masque le symptôme numérique mais ne restaure pas l'identifiabilité
    statistique. 'diag' ou 'spherical' restent identifiables même pour
    p >> n et sont recommandés ici ; 'full'/'tied' ne sont statistiquement
    défendables que sur un espace de features réduit (ex. axes PCA/PCoA,
    p < n).
    """
    log.info("run_gmm : chargement de la matrice…")
    matrix  = _load_matrix(args.matrix)
    samples = matrix.index.tolist()
    n       = len(samples)

    matrix, n_drop, n_total = _drop_nan_sites(matrix)
    if matrix.shape[1] == 0:
        log.error("Aucun site conservé après suppression des NaN. Arrêt.")
        sys.exit(1)

    p = matrix.shape[1]
    if args.covariance_type in ("full", "tied") and p >= n:
        log.warning(
            "  covariance_type='%s' avec p=%d features >= n=%d patients : "
            "la covariance est structurellement singulière (reg_covar ne "
            "compense qu'artificiellement). Envisager --covariance-type "
            "diag/spherical, ou fitter sur des coordonnées PCA/PCoA de "
            "dimension réduite plutôt que sur la matrice de sites complète.",
            args.covariance_type, p, n,
        )

    X = _get_scaler(args.use_robust_scaler).fit_transform(matrix.values)

    k_max   = min(args.k_max, n - 1)
    k_min   = min(args.k_min, k_max - 1)
    k_range = range(k_min, k_max + 1)

    log.info(
        "  calcul BIC pour k=%d à k=%d (covariance_type='%s', reg_covar=%.1e)…",
        k_min, k_max, args.covariance_type, args.reg_covar,
    )
    bics, aics, converged_flags = [], [], []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=args.covariance_type,
            reg_covar=args.reg_covar,
            random_state=args.random_state,
            n_init=10,
            max_iter=300,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        bics.append(bic)
        aics.append(aic)
        converged_flags.append(bool(gmm.converged_))
        log.info("  k=%d : BIC=%.1f  AIC=%.1f  converged=%s", k, bic, aic, gmm.converged_)
        if not gmm.converged_:
            log.warning(
                "  k=%d : non convergé en %d itérations — BIC potentiellement "
                "peu fiable pour ce k.", k, gmm.n_iter_,
            )

    k_final = args.k_final if args.k_final else int(np.argmin(bics)) + k_min
    log.info("  K final retenu (BIC minimal) : %d", k_final)

    gmm_final = GaussianMixture(
        n_components=k_final,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        random_state=args.random_state,
        n_init=20,
        max_iter=500,
    )
    labels = gmm_final.fit_predict(X)

    # Diagnostic clusters singletons — pertinent pour cette cohorte à
    # petits effectifs, cf. constat déjà fait sur d'autres analyses GMM.
    counts = pd.Series(labels).value_counts().sort_index()
    n_singletons = int((counts == 1).sum())
    if n_singletons > 0:
        log.warning(
            "  %d/%d clusters sont des singletons (1 seul patient) — "
            "cohérent avec un gradient continu plutôt que des sous-groupes "
            "discrets ; à interpréter avec prudence.",
            n_singletons, k_final,
        )

    os.makedirs(args.outdir, exist_ok=True)

    df_labels = pd.DataFrame({"sample": samples, "cluster": labels})
    df_labels.to_csv(
        os.path.join(args.outdir, f"gmm_labels{_suffix(args.tag)}.tsv"),
        sep="\t", index=False,
    )

    members_path = _write_cluster_membership(df_labels, args.outdir, f"gmm{_suffix(args.tag)}")
    log.info("  → %s", members_path)

    df_bic = pd.DataFrame({
        "k":         list(k_range),
        "bic":       bics,
        "aic":       aics,
        "converged": converged_flags,
    })
    df_bic.to_csv(
        os.path.join(args.outdir, f"gmm_bic{_suffix(args.tag)}.tsv"),
        sep="\t", index=False, float_format="%.6f",
    )

    # BIC/AIC plot — équivalent de l'elbow plot pour k-means
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), bics, "o-", color="#3B8BD4", linewidth=2, label="BIC")
    ax.plot(list(k_range), aics, "o--", color="#8EC5FF", linewidth=1.5, label="AIC")
    ax.axvline(k_final, color="#E8593C", linestyle="--", alpha=0.7,
               label=f"K retenu = {k_final}")
    ax.set_xlabel("Nombre de composantes K")
    ax.set_ylabel("Critère d'information (plus bas = meilleur)")
    ax.set_title(
        f"Sélection du K optimal — GMM méthylation (covariance='{args.covariance_type}')",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.outdir, f"gmm_bic{_suffix(args.tag)}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    log.info("  → gmm_bic%s.png", _suffix(args.tag))

    colors, patches = _color_vector(samples, None, None, labels)

    if args.pca_coords:
        pca_df = pd.read_csv(args.pca_coords, sep="\t", index_col=0).reindex(samples)
        _scatter_plot(
            coords=pca_df[["PC1", "PC2"]].values,
            samples=samples, colors=colors, patches=patches,
            xlabel="PC1", ylabel="PC2",
            title=f"PCA coloré par cluster GMM (K={k_final})",
            out_path=os.path.join(args.outdir, f"pca_scatter_gmm{_suffix(args.tag)}.png"),
            label_bool=args.label_bool,
        )

    if args.umap_coords:
        umap_df = pd.read_csv(args.umap_coords, sep="\t", index_col=0).reindex(samples)
        _scatter_plot(
            coords=umap_df[["UMAP1", "UMAP2"]].values,
            samples=samples, colors=colors, patches=patches,
            xlabel="UMAP1", ylabel="UMAP2",
            title=f"UMAP coloré par cluster GMM (K={k_final})",
            out_path=os.path.join(args.outdir, f"umap_scatter_gmm{_suffix(args.tag)}.png"),
            label_bool=args.label_bool,
        )

    log.info(
        "run_gmm : K=%d | distribution clusters :\n%s",
        k_final,
        df_labels["cluster"].value_counts().sort_index().to_string(),
    )

# ========================= STEP 6 : run_pairwise =======================

def _kmedoids(D: np.ndarray, k: int, n_init: int = 20,
              random_state: int = 42) -> tuple:
    """
    K-medoids (algorithme PAM simplifié) sur une matrice de dissimilarité.

    Contrairement à k-means dont les centroïdes sont des barycentres
    (points virtuels calculés comme moyennes — ce qui exige des valeurs
    complètes), les medoids sont des patients réels de la cohorte. Cela
    rend l'algorithme compatible avec des distances précalculées sur données
    incomplètes : aucune valeur n'est jamais inventée ou reconstituée à
    aucune étape.

    Algorithme :
      1. Initialisation aléatoire de k medoids parmi les patients.
      2. Affectation : chaque patient est assigné au medoid le plus proche
         selon D (distance nan-euclidienne).
      3. Mise à jour : pour chaque cluster, le nouveau medoid est le patient
         qui minimise la somme des distances intra-cluster (critère PAM).
      4. Répéter jusqu'à stabilité ou max_iter. Reprendre depuis n_init
         initialisations différentes et retenir la partition de coût minimal.

    Retourne (labels, cost, medoid_indices).
    """
    rng    = np.random.RandomState(random_state)
    n      = D.shape[0]
    best_labels  = None
    best_cost    = np.inf
    best_medoids = None

    # Remplacer les NaN dans D par une grande valeur finie pour les
    # comparaisons argmin (les paires sans co-observation ne seront jamais
    # choisies comme plus proches voisins, mais la structure du clustering
    # peut quand même les affecter à un cluster par défaut).
    D_safe = np.where(np.isnan(D), np.nanmax(D) * 10 + 1, D)
    np.fill_diagonal(D_safe, 0.0)

    for _ in range(n_init):
        medoids = rng.choice(n, k, replace=False)

        for _ in range(200):
            labels = np.argmin(D_safe[:, medoids], axis=1)

            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members) == 0:
                    continue
                sub_D = D_safe[np.ix_(members, members)]
                new_medoids[c] = members[np.argmin(sub_D.sum(axis=1))]

            if np.array_equal(new_medoids, medoids):
                break
            medoids = new_medoids

        cost = D_safe[np.arange(n), medoids[labels]].sum()
        if cost < best_cost:
            best_cost    = cost
            best_labels  = labels.copy()
            best_medoids = medoids.copy()

    return best_labels, best_cost, best_medoids


def cmd_run_pairwise(args):
    """
    Pipeline pairwise complet :
      1. Chargement de la matrice (NaN autorisés)
      2. Calcul de la matrice de dissimilarité nan-euclidienne
      3. Clustering hiérarchique (average ou complete linkage) + dendrogramme
      4. K-medoids sur k ∈ [k_min, k_max] + sélection par silhouette
      5. Fichiers de composition des clusters (hiérarchique + k-medoids)
      6. Scatter plots PCA/UMAP si coordonnées fournies
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform

    log.info("run_pairwise : chargement de %s…", args.matrix)
    matrix, samples = _load_matrix_with_nan(args.matrix)
    n       = len(samples)
    meta    = _load_metadata(getattr(args, "metadata", None), samples)

    M = matrix.astype(np.float64)

    # ── Matrice de dissimilarité ──────────────────────────
    log.info("  calcul de la matrice de dissimilarité (%s)…", args.distance_metric)

    if args.distance_metric == "euclidean":
        D, n_co = _pairwise_euclidean(M, args.min_coobs)
    elif args.distance_metric == "spearman":
        M_ranked = _rank_transform_available(M)
        D, n_co = _correlation_pairwise_available(M_ranked, metric_label="spearman")
    elif args.distance_metric == "pearson":
        D, n_co = _correlation_pairwise_available(M, metric_label="pearson")

    _log_coobs_diagnostic(n_co, args.min_coobs, n, metric_label=args.distance_metric)


    os.makedirs(args.outdir, exist_ok=True)

    # Sauvegarder la matrice de distance (format condensé + matrice complète)
    dist_path = os.path.join(args.outdir, f"distance_matrix{_suffix(args.tag)}.tsv")
    pd.DataFrame(D, index=samples, columns=samples).to_csv(
        dist_path, sep="\t", float_format="%.6f"
    )
    log.info("  → %s", dist_path)

    # ── Clustering hiérarchique ───────────────────────────
    log.info("  clustering hiérarchique (linkage='%s')…", args.linkage)
    condensed = squareform(D)
    Z = linkage(condensed, method=args.linkage)

    # Dendrogramme
    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), 8))
    dendrogram(
        Z, labels=samples, ax=ax,
        leaf_rotation=90, leaf_font_size=11,
        color_threshold=0.7 * max(Z[:, 2]),
    )

    # Ajuster l'axe Y pour éviter un dendrogramme écrasé lorsque la plus
    # petite dissimilarité de fusion est élevée. On place la limite
    # inférieure proche de la plus petite hauteur de fusion moins une marge
    # relative, sans descendre sous 0.
    try:
        if Z.shape[0] > 0:
            h_min = float(Z[:, 2].min())
            h_max = float(Z[:, 2].max())
            if np.isfinite(h_min) and np.isfinite(h_max) and h_max > h_min:
                margin = max((h_max - h_min) * 0.05, h_min * 0.01)
                y_bottom = max(0.0, h_min - margin)
                ax.set_ylim(bottom=y_bottom)
                log.info("  dendrogram y-axis ajustée : bottom=%.3f (min merge=%.3f)", y_bottom, h_min)
    except Exception:
        # Ne pas échouer le pipeline pour un problème d'affichage
        log.debug("  impossible d'ajuster l'axe Y du dendrogramme", exc_info=True)

    ax.set_title(
        f"Dendrogramme — clustering hiérarchique ({args.linkage} linkage, "
        f"distance {args.distance_metric})", fontweight="bold", fontsize=25,
    )
    ax.set_ylabel("Dissimilarité", fontsize=22)
    ax.tick_params(axis="y", labelsize=20)
    fig.tight_layout()
    dend_path = os.path.join(args.outdir, f"dendrogram{_suffix(args.tag)}.png")
    fig.savefig(dend_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", dend_path)

    # Partition si --n-cut demandé
    if args.n_cut is not None:
        hc_labels = fcluster(Z, t=args.n_cut, criterion="maxclust") - 1
        df_hc = pd.DataFrame({"sample": samples, "cluster": hc_labels})
        df_hc.to_csv(
            os.path.join(args.outdir, f"hierarchical_labels{_suffix(args.tag)}.tsv"),
            sep="\t", index=False,
        )
        hc_members_path = os.path.join(
            args.outdir, f"hierarchical_cluster_members{_suffix(args.tag)}.tsv"
        )
        _write_cluster_membership(df_hc, args.outdir,
                                  f"hierarchical{_suffix(args.tag)}")
        log.info(
            "  partition hiérarchique en %d clusters :\n%s",
            args.n_cut,
            df_hc["cluster"].value_counts().sort_index().to_string(),
        )

    # ── K-medoids ─────────────────────────────────────────
    k_max   = min(args.k_max, n - 1)
    k_min   = min(args.k_min, k_max - 1)
    k_range = range(k_min, k_max + 1)

    log.info("  k-medoids pour k=%d à k=%d…", k_min, k_max)

    # Silhouette sur la matrice D_fill (distances précalculées)
    from sklearn.metrics import silhouette_score as _sil

    results = []
    for k in k_range:
        labels, cost, medoids = _kmedoids(
            D, k, n_init=20, random_state=42,
        )
        sil = _sil(D, labels, metric="precomputed") if k > 1 else np.nan
        results.append((k, labels, cost, medoids, sil))
        log.info(
            "  k=%d : coût=%.2f  silhouette=%.4f  medoids=%s",
            k, cost, sil,
            [samples[m] for m in medoids],
        )

    # Sélection du k optimal (silhouette maximale, ou k_final forcé)
    if args.k_final is not None:
        k_best = args.k_final
        _, labels_best, _, medoids_best, _ = next(
            r for r in results if r[0] == k_best
        )
    else:
        sils = [(r[4], r[0], r[1], r[3]) for r in results if not np.isnan(r[4])]
        if not sils:
            log.error("Impossible de calculer la silhouette. Arrêt.")
            sys.exit(1)
        _, k_best, labels_best, medoids_best = max(sils, key=lambda x: x[0])

    log.info(
        "  K retenu (silhouette maximale) : %d  medoids : %s",
        k_best, [samples[m] for m in medoids_best],
    )

    df_km = pd.DataFrame({"sample": samples, "cluster": labels_best})
    df_km.to_csv(
        os.path.join(args.outdir, f"kmedoids_labels{_suffix(args.tag)}.tsv"),
        sep="\t", index=False,
    )
    _write_cluster_membership(df_km, args.outdir, f"kmedoids{_suffix(args.tag)}")

    # Tableau résumé silhouette / coût
    df_res = pd.DataFrame(
        [(r[0], r[2], r[4]) for r in results],
        columns=["k", "cost", "silhouette"],
    )
    df_res.to_csv(
        os.path.join(args.outdir, f"kmedoids_scores{_suffix(args.tag)}.tsv"),
        sep="\t", index=False, float_format="%.6f",
    )

    # Plot silhouette
    fig, ax = plt.subplots(figsize=(7, 4))
    ks   = [r[0] for r in results]
    sils = [r[4] for r in results]
    ax.plot(ks, sils, "o-", color="#1D9E75", linewidth=2)
    ax.axvline(k_best, color="#E8593C", linestyle="--", alpha=0.7,
               label=f"K retenu = {k_best}")
    ax.set_xlabel("Nombre de clusters K")
    ax.set_ylabel(f"Score silhouette (distance {args.distance_metric})")
    ax.set_title("K-medoids — sélection du K optimal", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.outdir, f"kmedoids_silhouette{_suffix(args.tag)}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # ── Scatter plots sur coordonnées existantes ──────────
    colors, patches = _color_vector(samples, None, None, labels_best)

    if args.pca_coords:
        pca_df = pd.read_csv(args.pca_coords, sep="\t", index_col=0).reindex(samples)
        _scatter_plot(
            coords=pca_df[["PC1", "PC2"]].values,
            samples=samples, colors=colors, patches=patches,
            xlabel="PC1", ylabel="PC2",
            title=f"PCA — clusters k-medoids pairwise (K={k_best})",
            out_path=os.path.join(
                args.outdir, f"pca_scatter_kmedoids{_suffix(args.tag)}.png"
            ),
            label_bool=args.label_bool,
        )

    if args.umap_coords:
        umap_df = pd.read_csv(args.umap_coords, sep="\t", index_col=0).reindex(samples)
        _scatter_plot(
            coords=umap_df[["UMAP1", "UMAP2"]].values,
            samples=samples, colors=colors, patches=patches,
            xlabel="UMAP1", ylabel="UMAP2",
            title=f"UMAP — clusters k-medoids pairwise (K={k_best})",
            out_path=os.path.join(
                args.outdir, f"umap_scatter_kmedoids{_suffix(args.tag)}.png"
            ),
            label_bool=args.label_bool,
        )

    log.info(
        "run_pairwise : K-medoids K=%d | distribution clusters :\n%s",
        k_best,
        df_km["cluster"].value_counts().sort_index().to_string(),
    )


# ========================= MAIN ==========================

def main():
    args = parse_args()
    dispatch = {
        "prepare":            cmd_prepare,
        "run_pca":            cmd_run_pca,
        "run_umap":           cmd_run_umap,
        "run_kmeans":         cmd_run_kmeans,
        "run_gmm":            cmd_run_gmm,
        "run_pairwise":       cmd_run_pairwise,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()