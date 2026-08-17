rule figures_bis:
    input:
        matrix = f"{config['dirs']['outdir_after']}/methylation_cohort.tsv"
    output:
        fig1  = f"{config['dirs']['outdir_after_figures']}/01_feature_label_distribution.png",
        fig2  = f"{config['dirs']['outdir_after_figures']}/02_heatmap_variable_sites.png",
        fig3  = f"{config['dirs']['outdir_after_figures']}/03_beta_distribution_by_feature.png",
        fig4a = f"{config['dirs']['outdir_after_figures']}/04a_beta_scatter_interpatient.png",
        fig4b = f"{config['dirs']['outdir_after_figures']}/04b_beta_scatter_by_origin.png"
    params:
        outdir  = config['dirs']['outdir_after_figures'],
        min_cov = config['min_cov']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_plot.py \
            --input   {input.matrix} \
            --outdir  {params.outdir} \
            --min-cov {params.min_cov}
        """