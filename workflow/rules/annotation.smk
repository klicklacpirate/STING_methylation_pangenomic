rule conversion_BED12:
    input:
        refgene= config['refs']['refgene']
    output:
        refgene_bed12 = f"{config['dirs']['dir_data']}/refGene/refGene.bed"
    shell:
        r"""
        awk -v c=3 -v st=4 -v ts=5 -v te=6 \
            -v ec=9 -v es=10 -v ee=11 -v n2=13 \
            'BEGIN{{OFS="\t"}} /^#/{{next}} {{
                chrom=$c; strand=$st; txStart=$ts; txEnd=$te;
                blockCount=$ec; geneName=$n2;
                split($es, starts, ","); split($ee, ends, ",");
                blockSizes=""; blockStarts="";
                for(i=1; i<=blockCount; i++){{
                    blockSizes  = blockSizes  (ends[i]-starts[i])          ",";
                    blockStarts = blockStarts (starts[i]-txStart+0)        ",";
                }}
                print chrom, txStart, txEnd, geneName, 0, strand, txStart, txEnd, 0, blockCount, blockSizes, blockStarts;
            }}' {input.refgene} > {output.refgene_bed12}
        """

rule extraction_exons:
    input:
        refgene_bed12 = rules.conversion_BED12.output.refgene_bed12
    output:
        refgene_exons_raw = f"{config['dirs']['dir_data']}/refGene/refGene_exons_raw.bed",
        refgene_exons     = f"{config['dirs']['dir_data']}/refGene/refGene_exons.bed"
    singularity:
        config["singularity"]["bedtools"]
    shell:
        """
        bedtools bed12tobed6 -i {input.refgene_bed12} > {output.refgene_exons_raw}
        sort -k1,1 -k2,2n {output.refgene_exons_raw} | bedtools merge -i - -c 4 -o distinct > {output.refgene_exons}
        """

rule extraction_introns:
    input:
        refgene_bed12 = rules.conversion_BED12.output.refgene_bed12,
        refgene_exons = rules.extraction_exons.output.refgene_exons
    output:
        refgene_introns = f"{config['dirs']['dir_data']}/refGene/refGene_introns.bed"
    singularity:
        config["singularity"]["bedtools"]
    shell:
        """
        bedtools subtract -a {input.refgene_bed12} -b {input.refgene_exons} > {output.refgene_introns}
        """

rule extraction_TSS:
    input:
        refgene_bed12 = rules.conversion_BED12.output.refgene_bed12
    output:
        refgene_tss = f"{config['dirs']['dir_data']}/refGene/refGene_TSS.bed"
    shell:
        """
        awk 'BEGIN{{OFS="\t"}} {{
            if ($6=="+") print $1, $2,   $2+1, $4, 0, $6;
            else         print $1, $3-1, $3,   $4, 0, $6;
        }}' {input.refgene_bed12} \
            | sort -k1,1 -k2,2n | uniq > {output.refgene_tss}
        """