import os
import csv

configfile: "/mnt/RetD/DNAseq_LR_AS/methylation_KL/SnakeMake_workflow/methylation_kl/config/config.yaml"


######## Load TSV sample table ########
data = []

with open(config["dataset"]) as f:
    for row in csv.reader(f, delimiter=" "):
        if row:
            data.append({
                "session": row[0],
                "flowcell": row[1],
                "barcode": row[2],
                "chemin_pod5": row[3],
                "BED_AS": row[4],
            })

######## Raccourcis utiles partout dans le workflow ########
SESSIONS    = [r["session"]      for r in data]
BARCODES    = [r["barcode"]      for r in data]

SESSION_BARCODES = [f"{r['session']}_{r['barcode']}" for r in data]

# Pour récupérer le chemin pod5 dans les lambdas
POD5_PATH = {
    (r["session"], r["barcode"]): r["chemin_pod5"]
    for r in data
}
# Pour récupérer le chemin du bam dans les lambdas
BAM_PATH = {
    (r["session"], r["barcode"]): os.path.join(
        config['dirs']['indir_bam'], r["session"], r["flowcell"], "alignment", "minimap2", f"{r['barcode']}.bam"
    )
    for r in data
}
# Pour récupérer le chemin du fichier bed pour la séparation AS dans les lambdas
NUMERO_BED_AS = {
    (r["session"], r["barcode"]): config["refs"][r["BED_AS"]]
    for r in data
}

DISTRIBUTION_BEFORE_OUTPUTS = [
    os.path.join(config['dirs']['outdir_before_figures'], "distribution_beta_full.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "distribution_beta_zoom.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "distribution_variance.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "distribution_coverage.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "01_feature_label_distribution.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "02_heatmap_variable_sites.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "03_beta_distribution_by_feature.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "04a_beta_scatter_interpatient.png"),
    os.path.join(config['dirs']['outdir_before_figures'], "04b_beta_scatter_by_origin.png")
]

FIGURES_AFTER_OUTPUTS = [
    os.path.join(config['dirs']['outdir_after_figures'], "01_feature_label_distribution.png"),
    os.path.join(config['dirs']['outdir_after_figures'], "02_heatmap_variable_sites.png"),
    os.path.join(config['dirs']['outdir_after_figures'], "03_beta_distribution_by_feature.png"),
    os.path.join(config['dirs']['outdir_after_figures'], "04a_beta_scatter_interpatient.png"),
    os.path.join(config['dirs']['outdir_after_figures'], "04b_beta_scatter_by_origin.png")
]

STATS_OUTPUTS = [
    os.path.join(config['dirs']['outdir_after_stats'], "pca_coords.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "pca_variance.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "pca_scatter.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "umap_coords.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "umap_scatter.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "kmeans_labels.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "kmeans_inertia.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "kmeans_elbow.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "pca_scatter_kmeans.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "umap_scatter_kmeans.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "gmm_labels.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "gmm_bic.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "pca_scatter_gmm.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "umap_scatter_gmm.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "distance_matrix.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "dendrogram.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "kmedoids_scores.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "kmedoids_labels.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "kmedoids_silhouette.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "pca_scatter_kmedoids.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "umap_scatter_kmedoids.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "pcoa_eigenvalues.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "pcoa_coordinates.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "pcoa_scree.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "permanova_run.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "permanova_permutation_run.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "pcoa_ordination.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "axis_risk_correlation.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "risk_gradient_loocv.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "permutation_test.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "risk_gradient_permutation.png"),
    os.path.join(config['dirs']['outdir_after_stats'], "risk_gradient_direction.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "risk_gradient_scores.tsv"),
    os.path.join(config['dirs']['outdir_after_stats'], "risk_gradient_projection.png")
]

######## Load rules ########
include: "rules/etapes_shell.smk"
include: "rules/annotation.smk"
include: "rules/etapes_python_before.smk"
include: "rules/plot_before.smk"
include: "rules/imputation.smk"
include: "rules/etapes_python_after.smk"
include: "rules/plot_after.smk"
include: "rules/stats.smk"


######## Target rule ########
rule all:
    input:
        expand(DISTRIBUTION_BEFORE_OUTPUTS),
        expand(FIGURES_AFTER_OUTPUTS),
        expand(STATS_OUTPUTS)