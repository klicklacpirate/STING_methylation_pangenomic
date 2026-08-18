"""
plot_methylation.py
-------------------
Visualisation de la matrice de méthylation annotée.

Figures produites :
  1. Histogramme de la répartition des feature_label
  2. Heatmap clusterisée des sites les plus variables (dendrogrammes patients + sites)
  3. Distribution des betas par feature_label (violin/boxplot)
  4a. Scatter beta patient vs patient — 9 paires en 3 groupes pivot sans chevauchement
  4b. Scatter beta patient vs patient coloré par origine CpG / EPIC

Usage :
  python plot_methylation.py --input ma_matrice.tsv [options]

Options :
  --input       Chemin vers le fichier TSV (obligatoire)
  --outdir      Dossier de sortie des figures (défaut : figures/)
  --top_n       Nombre de sites variables pour la heatmap (défaut : 50)
  --min_cov     Couverture minimale pour filtrer les sites (défaut : 5)
  --fmt         Format des figures : png, pdf, svg (défaut : png)
  --dpi         Résolution des figures en DPI (défaut : 150)
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

# ── Palette cohérente ──────────────────────────────────────────────────────────
PALETTE_FEATURE = {
    "Promoter":   "#5B4AE8",
    "Exon":       "#1D9E75",
    "Intron":     "#EF9F27",
    "Intergenic": "#888780",
    "TSS":        "#D85A30",
    "cCRE":       "#D4537E",
}
DEFAULT_COLOR = "#B4B2A9"

HEATMAP_CMAP = "RdYlBu_r"   # bleu = hypo, rouge = hyper — standard WGBS/ONT

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_matrix(path: str) -> pd.DataFrame:
    """Charge la matrice TSV avec gestion des encodages et types mixtes."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,          # lecture brute pour éviter les DtypeWarning
        low_memory=False,
        encoding="latin-1",
    )
    print(f"  Matrice chargée : {df.shape[0]:,} sites × {df.shape[1]} colonnes")
    return df


def detect_beta_cov_cols(df: pd.DataFrame):
    """
    Détecte automatiquement les colonnes beta_* et cov_* dans la matrice.
    Retourne (beta_cols, cov_cols, barcode_ids).
    """
    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    cov_cols  = [c for c in df.columns if c.startswith("cov_")]
    barcodes  = [c.replace("beta_", "") for c in beta_cols]
    return beta_cols, cov_cols, barcodes


def cast_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convertit les colonnes listées en float ; '.' → NaN."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace(".", np.nan), errors="coerce")
    return df


def apply_min_cov(df: pd.DataFrame, beta_cols: list, cov_cols: list,
                  min_cov: int) -> pd.DataFrame:
    """
    Masque les valeurs beta dont la couverture est < min_cov.
    Justification : un beta estimé sur < min_cov lectures a une variance
    binomiale élevée (var = p(1-p)/n), rendant la comparaison inter-patients
    peu fiable. On pose NaN pour exclure ces estimations non-robustes.
    """
    for beta_c, cov_c in zip(beta_cols, cov_cols):
        if cov_c in df.columns:
            mask = df[cov_c] < min_cov
            df.loc[mask, beta_c] = np.nan
    return df


def compute_variance(df: pd.DataFrame, beta_cols: list) -> pd.Series:
    """
    Variance inter-patients de la beta pour chaque site.
    Métrique standard pour identifier les DMRs (loci épigénétiquement variables).
    Sites avec un seul patient non-NaN → variance = 0 (exclus implicitement).
    """
    beta_matrix = df[beta_cols].values.astype(float)
    # ddof=1 : estimateur non biaisé (n-1) — pertinent pour petites cohortes
    variance = np.nanvar(beta_matrix, axis=1, ddof=1 if beta_matrix.shape[1] > 1 else 0)
    variance = np.where(np.isnan(variance), 0.0, variance)
    return pd.Series(variance, index=df.index, name="variance_beta")


def savefig(fig, outdir: str, name: str, fmt: str, dpi: int):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Figure 1 : histogramme feature_label ──────────────────────────────────────

def plot_feature_label(df: pd.DataFrame, outdir: str, fmt: str, dpi: int):
    if "feature_label" not in df.columns:
        print("  [SKIP] colonne 'feature_label' absente.")
        return

    counts = df["feature_label"].fillna("Unknown").value_counts()
    colors = [PALETTE_FEATURE.get(lbl, DEFAULT_COLOR) for lbl in counts.index]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5)

    # Annotations valeurs + pourcentages
    for bar, val in zip(bars, counts.values[::-1]):
        pct = val * 100.0 / total if total > 0 else 0.0
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f} %)", va="center", ha="left", fontsize=10, color="#444441")

    ax.set_xlabel("Nombre de sites CpG", fontsize=11)
    ax.set_title("Répartition des sites par annotation génomique", fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, counts.max() * 1.15)
    fig.tight_layout()

    savefig(fig, outdir, "01_feature_label_distribution", fmt, dpi)


# ── Figure 2 : heatmap clusterisée avec dendrogrammes ─────────────────────────

def plot_heatmap_variable_sites(df: pd.DataFrame, beta_cols: list, barcodes: list,
                                 top_n: int, outdir: str, fmt: str, dpi: int):
    """
    Heatmap avec clustering hiérarchique (dendrogrammes) sur les deux axes :
    - Axe sites (lignes) : clustering Ward sur distance euclidienne entre profils
      de méthylation → regroupe les CpGs co-régulés (même pattern inter-patients).
    - Axe patients (colonnes) : clustering Ward sur distance euclidienne entre
      patients → révèle les sous-groupes épigénétiques (potentiels sous-types cliniques).
    """
    variance = compute_variance(df, beta_cols)

    # Exclure les sites où moins de 2 patients ont une beta valide
    n_valid_patients = df[beta_cols].notna().sum(axis=1)
    variance = variance.where(n_valid_patients >= 2, other=0.0)

    n_valid = (variance > 0).sum()
    if n_valid == 0:
        print("  [SKIP] aucun site avec variance > 0 (données insuffisantes ou mono-patient).")
        return

    top_n = min(top_n, n_valid)
    top_integer_idx = variance.nlargest(top_n).index

    # Extraction de la sous-matrice beta
    sub = df.loc[top_integer_idx, beta_cols].copy().astype(float)
    if "site_id" in df.columns:
        sub.index = df.loc[top_integer_idx, "site_id"].values
    else:
        sub.index = top_integer_idx.astype(str)
    sub.columns = barcodes

    # ── Imputation par médiane de site (axe colonnes) ──────────────────────────
    # Justification : pdist requiert une matrice complète. L'imputation par médiane
    # préserve la tendance centrale du site sans extrapoler depuis d'autres patients.
    site_medians = sub.median(axis=1)
    sub_imputed = sub.apply(lambda col: col.fillna(site_medians))

    # ── Clustering hiérarchique ────────────────────────────────────────────────
    # Méthode Ward : minimise la variance intra-cluster → dendrogramme compact
    # et lisible, standard en épigénomique (utilisé par minfi, ChAMP, etc.)
    # Distance euclidienne : appropriée pour des beta [0,100] sur même échelle.

    # Clustering des sites (lignes) — on transpose pour avoir sites en lignes
    if sub_imputed.shape[0] > 1:
        link_rows = linkage(pdist(sub_imputed.values, metric="euclidean"), method="ward")
        row_order = leaves_list(link_rows)
    else:
        row_order = list(range(sub_imputed.shape[0]))
        link_rows = None

    # Clustering des patients (colonnes)
    if sub_imputed.shape[1] > 1:
        link_cols = linkage(pdist(sub_imputed.values.T, metric="euclidean"), method="ward")
        col_order = leaves_list(link_cols)
    else:
        col_order = list(range(sub_imputed.shape[1]))
        link_cols = None

    # Réordonnancement selon le clustering
    sub_clustered = sub.iloc[row_order, :].iloc[:, col_order]
    # On utilise sub (avec NaN d'origine) pour la heatmap : les NaN s'affichent
    # en gris, ce qui préserve l'information de données manquantes pour le lecteur.

    # ── Construction de la figure via clustermap de seaborn ───────────────────
    # clustermap gère nativement les dendrogrammes en bordure.
    norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)

    cell_h = max(0.18, min(0.40, 12 / top_n))
    fig_h  = max(5, top_n * cell_h + 3)
    fig_w  = max(4, len(barcodes) * 1.8 + 3)

    # Masque NaN pour affichage en gris
    mask_nan = sub_clustered.isna()

    cg = sns.clustermap(
        sub_clustered.fillna(-1),   # valeur sentinelle ; masquée par le mask
        row_linkage=link_rows if link_rows is not None else None,
        col_linkage=link_cols if link_cols is not None else None,
        cmap=HEATMAP_CMAP,
        norm=norm,
        mask=mask_nan,
        linewidths=0.3,
        linecolor="#E0E0E0",
        figsize=(fig_w, fig_h),
        cbar_pos=(0.02, 0.85, 0.03, 0.12),   # colorbar en haut à gauche
        dendrogram_ratio=(0.15, 0.12),         # (row_dendro, col_dendro) proportion
        yticklabels=True,
        xticklabels=True,
    )

    # Colorbar label
    cg.cax.set_ylabel("Beta (% méthylation)", rotation=270, labelpad=12, fontsize=9)

    # Titres et labels
    cg.ax_heatmap.set_title(
        f"Top {top_n} sites les plus variables — clustering hiérarchique (Ward, euclidien)",
        fontsize=11, fontweight="bold", pad=14
    )
    cg.ax_heatmap.set_xlabel("Patient (barcode)", fontsize=10)
    cg.ax_heatmap.set_ylabel("Site CpG", fontsize=10)
    cg.ax_heatmap.tick_params(axis="y", labelsize=max(5, min(9, 180 // top_n)))
    cg.ax_heatmap.tick_params(axis="x", labelsize=9, rotation=45)

    # Note d'imputation
    cg.fig.text(
        0.01, 0.01,
        "Dendrogrammes calculés après imputation par médiane de site (NaN → gris dans la heatmap).",
        fontsize=7, color="#888780", style="italic"
    )

    path = os.path.join(outdir, f"02_heatmap_variable_sites.{fmt}")
    os.makedirs(outdir, exist_ok=True)
    cg.fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(cg.fig)
    print(f"  → {path}")


# ── Figure 3 : distribution beta par feature_label ────────────────────────────

def plot_beta_by_feature(df: pd.DataFrame, beta_cols: list, barcodes: list,
                          outdir: str, fmt: str, dpi: int):
    if "feature_label" not in df.columns or not beta_cols:
        print("  [SKIP] colonnes manquantes pour violin plot.")
        return

    # Format long pour seaborn
    records = []
    for bc, bc_col in zip(barcodes, beta_cols):
        tmp = df[["feature_label", bc_col]].copy()
        tmp.columns = ["feature_label", "beta"]
        tmp["barcode"] = bc
        records.append(tmp)
    long_df = pd.concat(records, ignore_index=True).dropna(subset=["beta"])
    long_df["feature_label"] = long_df["feature_label"].fillna("Unknown")

    order = long_df.groupby("feature_label")["beta"].median().sort_values().index.tolist()
    palette = {lbl: PALETTE_FEATURE.get(lbl, DEFAULT_COLOR) for lbl in order}

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.4), 5))
    sns.violinplot(
        data=long_df, x="feature_label", y="beta", order=order,
        palette=palette, ax=ax, inner="quartile",
        cut=0, linewidth=0.7,
    )
    ax.set_xlabel("Annotation génomique", fontsize=11)
    ax.set_ylabel("Beta (% méthylation)", fontsize=11)
    ax.set_title("Distribution de la méthylation par annotation génomique", fontsize=12,
                 fontweight="bold", pad=10)
    ax.set_ylim(-5, 105)
    ax.axhline(50, color="#888", linestyle="--", linewidth=0.8, alpha=0.6, label="50 %")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()

    savefig(fig, outdir, "03_beta_distribution_by_feature", fmt, dpi)


# ── Figure 4 : scatter beta patient vs patient ─────────────────────────────────

def build_pivot_pairs(n_patients: int, n_groups: int = 3, per_group: int = 3) -> list:
    """
    Construit n_groups × per_group paires (pivot_i, target_j) sans chevauchement
    entre groupes : chaque groupe possède un patient pivot distinct ET des patients
    cibles distincts des autres groupes.

    Stratégie :
      - Les n_groups premiers indices sont les pivots (0, 1, 2).
      - Les patients restants sont distribués équitablement comme cibles.
      - Si le nombre de patients est insuffisant pour remplir tous les groupes,
        on recycle les cibles (avec avertissement) plutôt que de supprimer des plots.

    Retourne une liste plate de tuples (i, j) prête pour enumerate().
    """
    if n_patients < n_groups + 1:
        # Pas assez de patients : fallback sur toutes les paires disponibles
        from itertools import combinations
        return list(combinations(range(n_patients), 2))[: n_groups * per_group]

    pivots  = list(range(n_groups))
    targets = list(range(n_groups, n_patients))

    # Si pas assez de cibles uniques, on recycle
    if len(targets) < n_groups * per_group:
        import math
        repeats = math.ceil(n_groups * per_group / len(targets))
        targets = (targets * repeats)[: n_groups * per_group]
        print(f"  [INFO] Recyclage des cibles (seulement {n_patients} patients disponibles).")

    pairs = []
    for g in range(n_groups):
        pivot = pivots[g]
        group_targets = targets[g * per_group : (g + 1) * per_group]
        for t in group_targets:
            pairs.append((pivot, t))
    return pairs


def plot_beta_scatter(df: pd.DataFrame, beta_cols: list, barcodes: list,
                       outdir: str, fmt: str, dpi: int):
    """
    4 scatter plots organisés en 2 groupes pivot (2 lignes × 2 colonnes).
    Chaque ligne correspond à un patient pivot comparé à 2 patients cibles
    sans chevauchement entre lignes — visualisation lisible de la (dis)similarité
    épigénétique inter-individuelle.
    """
    if len(beta_cols) < 2:
        print("  [SKIP] scatter nécessite au moins 2 patients.")
        return

    N_GROUPS  = 2
    PER_GROUP = 2
    pairs = build_pivot_pairs(len(beta_cols), n_groups=N_GROUPS, per_group=PER_GROUP)

    ncols = PER_GROUP
    nrows = N_GROUPS
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.2),
                              squeeze=False)

    feature_col = df["feature_label"].fillna("Unknown") if "feature_label" in df.columns else None

    for idx, (i, j) in enumerate(pairs):
        row = idx // ncols
        col = idx % ncols
        ax  = axes[row][col]

        x = df[beta_cols[i]].astype(float)
        y = df[beta_cols[j]].astype(float)
        mask = x.notna() & y.notna()

        if feature_col is not None:
            for lbl, grp in df[mask].groupby(feature_col[mask]):
                ax.scatter(x[grp.index], y[grp.index],
                           color=PALETTE_FEATURE.get(lbl, DEFAULT_COLOR),
                           alpha=0.35, s=6, label=lbl, rasterized=True)
            if idx == 0:
                ax.legend(fontsize=7, markerscale=2, framealpha=0.6,
                          loc="upper left", title="Feature", title_fontsize=7)
        else:
            ax.scatter(x[mask], y[mask], alpha=0.3, s=5, color="#5B4AE8", rasterized=True)

        # Droite identité y = x
        ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.5)
        r = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 1 else np.nan
        ax.set_title(f"{barcodes[i]} vs {barcodes[j]}\nr = {r:.3f}", fontsize=10)
        ax.set_xlabel(f"Beta {barcodes[i]} (%)", fontsize=9)
        ax.set_ylabel(f"Beta {barcodes[j]} (%)", fontsize=9)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.spines[["top", "right"]].set_visible(False)

        # Étiquette de groupe (pivot) sur la première colonne de chaque ligne
        if col == 0:
            ax.set_ylabel(f"[Groupe {row+1}] Beta {barcodes[j]} (%)", fontsize=9)

    # Masquer les axes vides si moins de 9 paires
    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(
        "Corrélation inter-patients de la méthylation (CpG)\n"
        "4 paires — 2 patients pivot (lignes) × 2 patients cibles (colonnes), sans chevauchement",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    savefig(fig, outdir, "04a_beta_scatter_interpatient", fmt, dpi)


def classify_origin_category(df: pd.DataFrame) -> pd.Series:
    """Classe les sites selon l'origine CpG / EPIC dans la colonne 'origin'."""
    if "origin" not in df.columns:
        return pd.Series([], dtype=object)

    origin = df["origin"].astype(str).fillna("").str.strip()
    is_cpg = origin.str.contains(r"\bcpgIsland\b", case=False, na=False)
    is_epic = origin.str.contains(r"\b(450k|850k|v2)\b", case=False, na=False)

    category = np.full(len(df), "Other", dtype=object)
    category[is_cpg & is_epic] = "CpG island & EPIC"
    category[is_cpg & ~is_epic] = "CpG island"
    category[~is_cpg & is_epic] = "EPIC"

    return pd.Series(category, index=df.index, name="origin_category")


def plot_beta_scatter_by_origin(df: pd.DataFrame, beta_cols: list, barcodes: list,
                                 outdir: str, fmt: str, dpi: int):
    """
    Même organisation que 4a (2 groupes pivot × 2 cibles, 4 plots),
    mais les points sont colorés par catégorie d'origine génomique (CpG island / EPIC).
    """
    if len(beta_cols) < 2:
        print("  [SKIP] scatter 4b nécessite au moins 2 patients.")
        return
    if "origin" not in df.columns:
        print("  [SKIP] scatter 4b nécessite la colonne 'origin'.")
        return

    origin_category = classify_origin_category(df)
    if not origin_category.isin(["CpG island", "EPIC", "CpG island & EPIC"]).any():
        print("  [SKIP] aucune ligne classée en CpG island / EPIC / les deux.")
        return

    palette = {
        "CpG island": "#5B4AE8",
        "EPIC": "#D85A30",
        "CpG island & EPIC": "#1D9E75",
        "Other": "#B4B2A9",
    }

    N_GROUPS  = 2
    PER_GROUP = 2
    pairs = build_pivot_pairs(len(beta_cols), n_groups=N_GROUPS, per_group=PER_GROUP)

    ncols = PER_GROUP
    nrows = N_GROUPS
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.2),
                              squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        row = idx // ncols
        col = idx % ncols
        ax  = axes[row][col]

        x = df[beta_cols[i]].astype(float)
        y = df[beta_cols[j]].astype(float)
        mask = x.notna() & y.notna()

        grouped = df[mask].groupby(origin_category[mask])
        for cat, grp in grouped:
            color = palette.get(cat, palette["Other"])
            ax.scatter(
                x.loc[grp.index],
                y.loc[grp.index],
                color=color,
                alpha=0.35,
                s=6,
                rasterized=True,
            )

        ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.5)
        r = np.corrcoef(x[mask], y[mask])[0, 1] if mask.sum() > 1 else np.nan
        ax.set_title(f"{barcodes[i]} vs {barcodes[j]}\nr = {r:.3f}", fontsize=10)
        ax.set_xlabel(f"Beta {barcodes[i]} (%)", fontsize=9)
        ax.set_ylabel(f"Beta {barcodes[j]} (%)", fontsize=9)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.spines[["top", "right"]].set_visible(False)

        if idx == 0:
            legend_handles = [
                plt.Line2D([0], [0], marker="o", color="w", label="CpG island",
                           markerfacecolor=palette["CpG island"], markersize=6, alpha=0.7),
                plt.Line2D([0], [0], marker="o", color="w", label="EPIC",
                           markerfacecolor=palette["EPIC"], markersize=6, alpha=0.7),
                plt.Line2D([0], [0], marker="o", color="w", label="CpG island + EPIC",
                           markerfacecolor=palette["CpG island & EPIC"], markersize=6, alpha=0.7),
            ]
            ax.legend(handles=legend_handles, fontsize=8, framealpha=0.6,
                      loc="upper left", title="Origine")

    for k in range(len(pairs), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle(
        "Corrélation inter-patients de la méthylation (CpG) — coloré par origine CpG / EPIC\n"
        "4 paires — 2 patients pivot (lignes) × 2 patients cibles (colonnes), sans chevauchement",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    savefig(fig, outdir, "04b_beta_scatter_by_origin", fmt, dpi)

# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",   required=True,  help="Matrice TSV annotée")
    p.add_argument("--outdir",  default="figures", help="Dossier de sortie (défaut : figures/)")
    p.add_argument("--top-n",   type=int, default=50,  help="Sites variables pour heatmap (défaut : 50)")
    p.add_argument("--min-cov", type=int, default=5,  help="Couverture minimale (défaut : 5)")
    p.add_argument("--fmt",     default="png", choices=["png", "pdf", "svg"],
                   help="Format de sortie (défaut : png)")
    p.add_argument("--dpi",     type=int, default=150, help="Résolution PNG (défaut : 150)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  plot_methylation.py")
    print(f"{'='*60}")
    print(f"  Fichier  : {args.input}")
    print(f"  Sortie   : {args.outdir}/")
    print(f"  min_cov  : {args.min_cov}")
    print(f"  top_n    : {args.top_n}")
    print(f"  format   : {args.fmt} @ {args.dpi} dpi")
    print(f"{'='*60}\n")

    # 1. Chargement
    print("[1/8] Chargement de la matrice...")
    df = load_matrix(args.input)

    # 2. Détection des colonnes beta/cov
    print("[2/8] Détection des colonnes beta / couverture...")
    beta_cols, cov_cols, barcodes = detect_beta_cov_cols(df)
    if not beta_cols:
        sys.exit("  ERREUR : aucune colonne beta_* trouvée. Vérifiez le format du fichier.")
    print(f"  Patients détectés ({len(barcodes)}) : {', '.join(barcodes)}")

    # 3. Cast numérique + filtre couverture
    print("[3/8] Conversion numérique et filtre couverture...")
    df = cast_numeric(df, beta_cols + cov_cols)
    df = apply_min_cov(df, beta_cols, cov_cols, args.min_cov)
    n_valid = df[beta_cols].notna().all(axis=1).sum()
    print(f"  Sites avec beta valide pour tous les patients : {n_valid:,}")

    # 4. Figures
    print("[4/8] Figure 1 — Répartition feature_label...")
    plot_feature_label(df, args.outdir, args.fmt, args.dpi)

    print("[5/8] Figure 2 — Heatmap clusterisée avec dendrogrammes...")
    plot_heatmap_variable_sites(df, beta_cols, barcodes, args.top_n,
                                 args.outdir, args.fmt, args.dpi)

    print("[6/8] Figure 3 — Distribution beta par feature...")
    plot_beta_by_feature(df, beta_cols, barcodes, args.outdir, args.fmt, args.dpi)

    print("[7/8] Figure 4a — Scatter inter-patients (9 paires, 3 groupes pivot)...")
    plot_beta_scatter(df, beta_cols, barcodes, args.outdir, args.fmt, args.dpi)

    print("[8/8] Figure 4b — Scatter inter-patients coloré par origine CpG / EPIC...")
    plot_beta_scatter_by_origin(df, beta_cols, barcodes, args.outdir, args.fmt, args.dpi)


    print(f"\n✓ Terminé. Figures dans : {os.path.abspath(args.outdir)}/\n")


if __name__ == "__main__":
    main()