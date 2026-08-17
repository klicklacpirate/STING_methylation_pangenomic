rule parse_sample:
    input:
        pileup = os.path.join(config['dirs']['cache'], "{session}", "{session}_{barcode}.bed")
    output:
        parquet = os.path.join(config['dirs']['cache'], "{session}", "{session}_{barcode}.parquet")
    threads: 12
    resources:
        mem_mb = 10000
    params:
        outdir = lambda wildcards: os.path.join(config['dirs']['cache'], wildcards.session),
        sample = lambda wildcards: f"{wildcards.session}_{wildcards.barcode}"
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py parse_sample \
            --input       {input.pileup} \
            --sample-name {params.sample} \
            --outdir      {params.outdir} \
            --etape       before_imputation
        """

rule build_cohort_matrix:
    input:
        pileup_file = expand(
            os.path.join(config['dirs']['cache'], "{session}", "{session}_{barcode}.parquet"),
            zip,
            session=SESSIONS,
            barcode=BARCODES,
        )
    output:
        parquet        = os.path.join(config['dirs']['cache'], "cohort_matrix.parquet"),
        excluded_sites = os.path.join(config['dirs']['cache'], "excluded_sites_from_sites_file.bed")
    threads:
        12
    resources:
        mem_mb = 40000
    params:
        outdir       = os.path.join(config['dirs']['cache']),
        min_coverage = config['min_cov'],
        sample_names = " ".join(SESSION_BARCODES),
        sites        = config['refs']['cpg_interet']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py build_cohort_matrix \
        --input        {input.pileup_file} \
        --sample-names {params.sample_names} \
        --sites-file   {params.sites} \
        --min-cov      {params.min_coverage} \
        --outdir       {params.outdir} \
        """

rule assign_origins:
    input:
        cohort_matrix   = rules.build_cohort_matrix.output.parquet,
        sites_interet = config['refs']['cpg_interet']
    output:
        cohort_origin = os.path.join(config['dirs']['cache'], "cohort_with_origins.parquet")
    params:
        outdir = os.path.join(config['dirs']['cache'])
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py assign_origins \
          --cohort-matrix {input.cohort_matrix} \
          --source-bed    {input.sites_interet} \
          --outdir        {params.outdir} \
        """

rule annotate_genomic:
    input:
        cohort_origin   = rules.assign_origins.output.cohort_origin,
        refgene_tss     = os.path.join(config['dirs']['dir_data'], "refGene", "refGene_TSS.bed"),
        refgene_exons   = os.path.join(config['dirs']['dir_data'], "refGene", "refGene_exons.bed"),
        refgene_introns = os.path.join(config['dirs']['dir_data'], "refGene", "refGene_introns.bed"),
        ccre            = config['refs']['ccre']
    output:
        cohort_annotated_refgene = os.path.join(config['dirs']['cache'], "cohort_genomic.parquet")
    params:
        outdir = os.path.join(config['dirs']['cache'])
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

rule export_tsv:
    input:
        cohort = rules.annotate_genomic.output.cohort_annotated_refgene
    output:
        matrice = os.path.join(config["dirs"]["outdir_before"], "methylation_cohort.tsv")
    params:
        outdir = config['dirs']['outdir_before']
    singularity:
        config["singularity"]["python-tools"]
    shell:
        """
        python3 workflow/scripts/methylation_annotate.py export_tsv \
          --cohort-genomic  {input.cohort} \
          --outdir          {params.outdir} \
        """