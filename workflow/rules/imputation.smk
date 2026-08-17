rule constat:
    input:
        cohort = os.path.join(config['dirs']['outdir_before'], "methylation_cohort.tsv")
    output:
        cov_summary  = os.path.join(config['dirs']['outdir_before'], "coverage_summary.csv"),
        cov_per_site = os.path.join(config['dirs']['outdir_before'], "per_site_coverage.csv"),
        site_75pct   = os.path.join(config['dirs']['outdir_before'], "coverage_distribution_covered_at_least_75pct_before_imputation.bed")
    params:
        outdir = os.path.join(config['dirs']['outdir_before']),
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/perte_info_analyse.py \
            --input  {input.cohort} \
            --outdir {params.outdir}
        """

rule gimmecpg:
    input:
        pileup_file = expand(
            os.path.join(config['dirs']['cache'], "{session}", "{session}_{barcode}.bed"),
            zip,
            session=SESSIONS,
            barcode=BARCODES,
        )
    output:
        pileup_imputed     = expand(
            os.path.join(config['dirs']['gimmecpg'], "{session}_{barcode}.bed"),
            zip,
            session=SESSIONS,
            barcode=BARCODES,
        ),
        nan_summary_global = os.path.join(config['dirs']['gimmecpg'], "imputation_nan_summary_global.tsv"),
        nan_summary        = os.path.join(config['dirs']['gimmecpg'], "imputation_nan_summary.csv")
    params:
        input_dir    = config['dirs']['cache'],
        outdir       = os.path.join(config['dirs']['gimmecpg']),
        min_coverage = config['min_cov'],
        cpg_ref      = config['refs']['cpg_interet']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 /mnt/RetD/DNAseq_LR_AS/methylation_KL/GIMMEcpg/gimmecpg-python/gimmecpg_python/main.py \
          --input  {params.input_dir} \
          --outdir {params.outdir} \
          --ref    {params.cpg_ref} \
          --minCov {params.min_coverage}
        """