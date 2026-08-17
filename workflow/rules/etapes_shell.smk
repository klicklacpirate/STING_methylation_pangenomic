rule extraction_readnames:
    input:
        cpg        = config['refs']['cpg_interet'],
        bam_aligne = lambda wildcards: BAM_PATH[(wildcards.session, wildcards.barcode)]
    output:
        readname = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_readnames.txt")
    params:
        threads = config['threads']
    singularity:
       config["singularity"]["samtools"]
    shell:
        """
        samtools view -@ {params.threads} -F 0x904 -L {input.cpg} {input.bam_aligne} \
        | cut -f1 | tr -d '\r' | sort -u \
        > {output}
        """

rule dorado_basecall:
    input:
        genome   = config['refs']['genome'],
        pod5_dir = lambda wildcards: POD5_PATH[(wildcards.session, wildcards.barcode)],
        readname = rules.extraction_readnames.output.readname
    output:
        bam = os.path.join(config['dirs']['cache'], "{session}", "{barcode}.bam")
    resources:
        gpu = 1
    params:
        threads   = config['threads'],
        model     = "/models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
        model_mod = "/models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v3",
        dorado    = config["singularity"]["dorado"]
    shell:
        """
        singularity exec --nv {params.dorado} dorado basecaller \
        {params.model} {input.pod5_dir} \
        -x cuda:all \
        --read-ids {input.readname} \
        --reference {input.genome} \
        --modified-bases-models {params.model_mod} \
        > {output.bam}
        """

rule filtre_MAPQ:
    input:
        bam = rules.dorado_basecall.output.bam
    output:
        bam = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_filtered.bam")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["samtools"]
    shell:
        """
        samtools view -@ {params.threads} -b -q 1 {input.bam} > {output.bam}
        """

rule trier:
    input:
        rules.filtre_MAPQ.output.bam
    output:
        bam = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_sorted.bam"),
        bai = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_sorted.bam.bai")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["samtools"]
    shell:
        """
        samtools sort -@ {params.threads} -o {output.bam} {input}
        samtools index -@ {params.threads} {output.bam}
        """

rule separation_AS:
    input:
        bam = rules.trier.output.bam,
        bed = lambda wildcards: NUMERO_BED_AS[(wildcards.session, wildcards.barcode)]
    output:
        bam_antitarget = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_antitarget.bam"),
        bam_target     = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_target.bam")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["samtools"]
    shell:
        """
        samtools view -@ {params.threads} \
        -b \
        -L {input.bed} \
        -U {output.bam_antitarget} \
        {input.bam} > {output.bam_target}
        """

rule down_sampling:
    input:
        bam = rules.separation_AS.output.bam_target
    output:
        bam_ds = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_target_ds.bam")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["samtools"]
    shell:
        """
        samtools view -@ {params.threads} -b -s 100.125 {input.bam} > {output.bam_ds}
        """

rule detect_hyper_cov:
    input:
        bam = rules.separation_AS.output.bam_antitarget
    output:
        bed_highcov = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_highcov.bed")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["bedtools"]
    shell:
        """
        bedtools genomecov -ibam {input.bam} -bga \
        | awk '$4 > 100 {{print $1"\t"$2"\t"$3}}' \
        | bedtools merge > {output.bed_highcov}
        """

rule filter_hyper_cov:
    input:
        bam         = rules.separation_AS.output.bam_antitarget,
        bed_highcov = rules.detect_hyper_cov.output.bed_highcov
    output:
        bam_antitarget_filtered = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_antitarget_filtered.bam"),
        read_highcov            = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_highcov_reads.bam")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["samtools"]
    shell:
        """
        samtools view -@ {params.threads} -b -L {input.bed_highcov} -U {output.bam_antitarget_filtered} {input.bam} > {output.read_highcov}
        """

rule merge:
    input:
        bam_antitarget = rules.filter_hyper_cov.output.bam_antitarget_filtered,
        bam_target_ds  = rules.down_sampling.output.bam_ds
    output:
        bam = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_merged.bam")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["samtools"]
    shell:
        """
        samtools merge -@ {params.threads} -f {output.bam} {input.bam_target_ds} {input.bam_antitarget}
        """

rule sort_merge:
    input:
        bam = rules.merge.output.bam
    output:
        bam = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_merged_sorted.bam"),
        bai = os.path.join(config['dirs']['cache'], "{session}", "{barcode}_merged_sorted.bam.bai")
    params:
        threads = config['threads']
    singularity:
        config["singularity"]["samtools"]
    shell:
        """
        samtools sort -@ {params.threads} -o {output.bam} {input.bam}
        samtools index -@ {params.threads} {output.bam}
        """

rule modkit_pileup:
    input:
        rules.sort_merge.output.bam
    output:
        bed = os.path.join(config['dirs']['cache'], "{session}", "{session}_{barcode}.bed"),
        log = os.path.join(config['dirs']['cache'], "{session}", "{session}_{barcode}.log")
    resources:
        #gpu = 1
        mem_mb = 40000
    singularity:
        config["singularity"]["modkit"]
    shell:
        """
        modkit pileup \
        {input} {output.bed} \
        --log-filepath {output.log}
        """