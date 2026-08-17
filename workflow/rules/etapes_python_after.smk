rule parse_sample_bis:
    input:
        pileup_filtered = os.path.join(config['dirs']['gimmecpg'], "{session}_{barcode}.bed")
    output:
        parquet = os.path.join(config['dirs']['gimmecpg'], "{session}_{barcode}.parquet")
    threads: 12
    resources:
        mem_mb = 10000
    params:
        outdir = os.path.join(config['dirs']['gimmecpg']),
        sample = lambda wildcards: f"{wildcards.session}_{wildcards.barcode}"
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py parse_sample \
            --input       {input.pileup_filtered} \
            --sample-name {params.sample} \
            --outdir      {params.outdir}
            --etape       after_imputation
        """

rule build_cohort_matrix_bis:
    input:
        pileup_file = expand(
            os.path.join(config['dirs']['gimmecpg'], "{session}_{barcode}.parquet"),
            zip,
            session=SESSIONS,
            barcode=BARCODES,
        )
    output:
        parquet = os.path.join(config['dirs']['gimmecpg'], "cohort_matrix.parquet")
    threads: 12
    resources:
        mem_mb = 40000
    params:
        outdir       = os.path.join(config['dirs']['gimmecpg']),
        min_coverage = config['min_cov'],
        sample_names = " ".join(SESSION_BARCODES),
        sites_75     = os.path.join(config['dirs']['outdir_before'], "coverage_distribution_covered_at_least_75pct_before_imputation.bed")
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py build_cohort_matrix \
        --input        {input.pileup_file} \
        --sample-names {params.sample_names} \
        --sites-file   {params.sites_75} \
        --min-cov      {params.min_coverage} \
        --outdir       {params.outdir} \
        """

rule assign_origins_bis:
    input:
        cohort_matrix   = rules.build_cohort_matrix_bis.output.parquet,
        sites_interet = config['refs']['cpg_interet']
    output:
        cohort_origin = os.path.join(config['dirs']['gimmecpg'], "cohort_with_origins.parquet")
    params:
        outdir = os.path.join(config['dirs']['gimmecpg'])
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py assign_origins \
          --cohort-matrix {input.cohort_matrix} \
          --source-bed    {input.sites_interet} \
          --outdir        {params.outdir} \
        """

rule annotate_genomic_bis:
    input:
        cohort_origin   = rules.assign_origins_bis.output.cohort_origin,
        refgene_tss     = os.path.join(config['dirs']['dir_data'], "refGene", "refGene_TSS.bed"),
        refgene_exons   = os.path.join(config['dirs']['dir_data'], "refGene", "refGene_exons.bed"),
        refgene_introns = os.path.join(config['dirs']['dir_data'], "refGene", "refGene_introns.bed"),
        ccre            = config['refs']['ccre']
    output:
        cohort_annotated_refgene = os.path.join(config['dirs']['gimmecpg'], "cohort_genomic.parquet")
    params:
        outdir = os.path.join(config['dirs']['gimmecpg'])
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py annotate_genomic \
          --cohort-with-origins  {input.cohort_origin} \
          --tss-bed              {input.refgene_tss} \
          --exons-bed            {input.refgene_exons} \
          --introns-bed          {input.refgene_introns} \
          --ccre-bed             {input.ccre} \
          --outdir               {params.outdir} \
        """

rule export_tsv_bis:
    input:
        cohort = rules.annotate_genomic_bis.output.cohort_annotated_refgene
    output:
        matrice = os.path.join(config["dirs"]["outdir_after"], "methylation_cohort.tsv")
    params:
        outdir = config['dirs']['outdir_after']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py export_tsv \
          --cohort-genomic  {input.cohort} \
          --outdir          {params.outdir} \
        """