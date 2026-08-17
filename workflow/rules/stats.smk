rule prepare:
    input:
        matrix_cohort = f"{config['dirs']['outdir_after']}/methylation_cohort.tsv"
    output:
        prepared_matrix = f"{config['dirs']['outdir_after_stats']}/clustering_matrix.parquet"
    params:
        outdir             = config['dirs']['outdir_after_stats'],
        methode_imputation = config["methode_imputation"]
    resources:
        mem_mb = 8000
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/clustering.py prepare \
            --cohort-tsv        {input.matrix_cohort} \
            --min-completeness  1 \
            --imputation-method {params.methode_imputation} \
            --outdir            {params.outdir}
        """

rule run_pca:
    input:
        prepared_matrix = rules.prepare.output.prepared_matrix
    output:
        pca_coords    = f"{config['dirs']['outdir_after_stats']}/pca_coords.tsv",
        pca_variance  = f"{config['dirs']['outdir_after_stats']}/pca_variance.tsv",
        pca_scatter   = f"{config['dirs']['outdir_after_stats']}/pca_scatter.png"
    params:
        outdir = config['dirs']['outdir_after_stats']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/clustering.py run_pca \
            --matrix       {input.prepared_matrix} \
            --use-robust-scaler \
            --outdir       {params.outdir}
        """

rule run_umap:
    input:
        prepared_matrix = rules.prepare.output.prepared_matrix
    output:
        umap_coords  = f"{config['dirs']['outdir_after_stats']}/umap_coords.tsv",
        umap_scatter = f"{config['dirs']['outdir_after_stats']}/umap_scatter.png"
    params:
        outdir = config['dirs']['outdir_after_stats'],
        dist   = 0.1,
        neigh  = 5
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/clustering.py run_umap \
            --matrix      {input.prepared_matrix} \
            --n-neighbors {params.neigh} \
            --min-dist    {params.dist} \
            --use-robust-scaler \
            --outdir      {params.outdir}
        """

rule run_kmeans:
    input:
        prepared_matrix = rules.prepare.output.prepared_matrix,
        pca_coords      = rules.run_pca.output.pca_coords,
        umap_coords     = rules.run_umap.output.umap_coords
    output:
        kmeans_labels  = f"{config['dirs']['outdir_after_stats']}/kmeans_labels.tsv",
        kmeans_inertia = f"{config['dirs']['outdir_after_stats']}/kmeans_inertia.tsv",
        kmeans_elbow   = f"{config['dirs']['outdir_after_stats']}/kmeans_elbow.png",
        kmeans_pca     = f"{config['dirs']['outdir_after_stats']}/pca_scatter_kmeans.png",
        kmeans_umap    = f"{config['dirs']['outdir_after_stats']}/umap_scatter_kmeans.png"
    params:
        outdir = config['dirs']['outdir_after_stats']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/clustering.py run_kmeans \
            --matrix      {input.prepared_matrix} \
            --pca-coords  {input.pca_coords} \
            --umap-coords {input.umap_coords} \
            --use-robust-scaler \
            --outdir      {params.outdir}
        """

rule run_gmm:
    input:
        prepared_matrix = rules.prepare.output.prepared_matrix,
        pca_coords      = rules.run_pca.output.pca_coords,
        umap_coords     = rules.run_umap.output.umap_coords
    output:
        gmm_labels  = f"{config['dirs']['outdir_after_stats']}/gmm_labels.tsv",
        gmm_bic     = f"{config['dirs']['outdir_after_stats']}/gmm_bic.tsv",
        gmm_pca     = f"{config['dirs']['outdir_after_stats']}/pca_scatter_gmm.png",
        gmm_umap    = f"{config['dirs']['outdir_after_stats']}/umap_scatter_gmm.png"
    params:
        outdir   = config['dirs']['outdir_after_stats'],
        cov_type = "spherical"
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/clustering.py run_gmm \
            --matrix          {input.prepared_matrix} \
            --pca-coords      {input.pca_coords} \
            --umap-coords     {input.umap_coords} \
            --covariance-type {params.cov_type} \
            --use-robust-scaler \
            --outdir          {params.outdir} \
        """

rule run_pairwise:
    input:
        prepared_matrix = rules.prepare.output.prepared_matrix,
        pca_coords      = rules.run_pca.output.pca_coords,
        umap_coords     = rules.run_umap.output.umap_coords
    output:
        matrix_distance  = f"{config['dirs']['outdir_after_stats']}/distance_matrix.tsv",
        dendrogram       = f"{config['dirs']['outdir_after_stats']}/dendrogram.png",
        scores           = f"{config['dirs']['outdir_after_stats']}/kmedoids_scores.tsv",
        labels           = f"{config['dirs']['outdir_after_stats']}/kmedoids_labels.tsv",
        silouette        = f"{config['dirs']['outdir_after_stats']}/kmedoids_silhouette.png",
        kmedoids_pca     = f"{config['dirs']['outdir_after_stats']}/pca_scatter_kmedoids.png",
        kmedoids_umap    = f"{config['dirs']['outdir_after_stats']}/umap_scatter_kmedoids.png"
    params:
        outdir          = config['dirs']['outdir_after_stats'],
        distance_metric = "spearman"
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/clustering.py run_pairwise \
            --matrix          {input.prepared_matrix} \
            --distance-metric {params.distance_metric} \
            --pca-coords      {input.pca_coords} \
            --umap-coords     {input.umap_coords} \
            --outdir          {params.outdir} \
        """

rule pcoa:
    input:
        matrix_distance = rules.run_pairwise.output.matrix_distance
    output:
        eigenvalues               = f"{config['dirs']['outdir_after_stats']}/pcoa_eigenvalues.tsv",
        pcoa_coords               = f"{config['dirs']['outdir_after_stats']}/pcoa_coordinates.tsv",
        scree                     = f"{config['dirs']['outdir_after_stats']}/pcoa_scree.png",
        permanova                 = f"{config['dirs']['outdir_after_stats']}/permanova_run.tsv",
        permanova_permutation     = f"{config['dirs']['outdir_after_stats']}/permanova_permutation_run.png",
        ordination                = f"{config['dirs']['outdir_after_stats']}/pcoa_ordination.png",
        axis_risk_correlation     = f"{config['dirs']['outdir_after_stats']}/axis_risk_correlation.tsv",
        risk_gradient             = f"{config['dirs']['outdir_after_stats']}/risk_gradient_loocv.tsv",
        permutation_test          = f"{config['dirs']['outdir_after_stats']}/permutation_test.tsv",
        risk_gradient_permutation = f"{config['dirs']['outdir_after_stats']}/risk_gradient_permutation.png",
        risk_gradient_direction   = f"{config['dirs']['outdir_after_stats']}/risk_gradient_direction.tsv",
        risk_gradient_scores      = f"{config['dirs']['outdir_after_stats']}/risk_gradient_scores.tsv",
        risk_gradient_projection  = f"{config['dirs']['outdir_after_stats']}/risk_gradient_projection.png"
    params:
        outdir = config['dirs']['outdir_after_stats'],
        risk   = config['refs']['risk'],
        metadata = config['refs']['metadata']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/risk_gradient.py \
            --distance-matrix  {input.matrix_distance} \
            --risk-file        {params.risk} \
            --metadata         {params.metadata} \
            --color-by         run \
            --permanova-column run \
            --outdir           {params.outdir} \
        """