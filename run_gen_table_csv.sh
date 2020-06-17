#!/bin/bash

set -e
#set -x

file_path=$1
basename_file_path=`basename $file_path`

new_suffix=".data.txt"
str_suffix=".extended.csv"
files=$(ls $file_path)
for filename in $files
do
	name=$(ls ${filename}|xargs basename | cut -d. -f1)
	echo $filename | grep $str_suffix\$    || no_correct_file_suffix_error

	if [[ $filename =~ $str_suffix ]]
	then
		cat $filename |sed 's/,/\t/g' |awk '$2=="*"' |grep  'GRCh38_AllHomopolymers_gt6bp_imperfectgt10bp_slop5|GRCh38_AllTandemRepeats_201to10000bp_slop5|GRCh38_AllTandemRepeats_51to200bp_slop5|GRCh38_AllTandemRepeats_gt10000bp_slop5|GRCh38_AllTandemRepeats_gt100bp_slop5|GRCh38_AllTandemRepeats_lt51bp_slop5|GRCh38_AllTandemRepeatsandHomopolymers_slop5|GRCh38_BadPromoters|GRCh38_HG001_GIABv3.2.2_compoundhet_slop50|GRCh38_HG001_GIABv3.2.2_varswithin50bp|GRCh38_HG001_GIABv3.3.2_RTG_PG_v3.3.2_SVs_alldifficultregions|GRCh38_HG001_GIABv3.3.2_RTG_PG_v3.3.2_SVs_notin_alldifficultregions|GRCh38_HG001_GIABv3.3.2_comphetindel10bp_slop50|GRCh38_HG001_GIABv3.3.2_comphetsnp10bp_slop50|GRCh38_HG001_GIABv3.3.2_complexandSVs|GRCh38_HG001_GIABv3.3.2_complexindel10bp_slop50|GRCh38_HG001_GIABv3.3.2_snpswithin10bp_slop50|GRCh38_HG001_PG2016-1.0_comphetindel10bp_slop50|GRCh38_HG001_PG2016-1.0_comphetsnp10bp_slop50|GRCh38_HG001_PG2016-1.0_complexindel10bp_slop50|GRCh38_HG001_PG2016-1.0_snpswithin10bp_slop50|GRCh38_HG001_PacBio_MetaSV|GRCh38_HG001_RTG_37.7.3_comphetindel10bp_slop50|GRCh38_HG001_RTG_37.7.3_comphetsnp10bp_slop50|GRCh38_HG001_RTG_37.7.3_complexindel10bp_slop50|GRCh38_HG001_RTG_37.7.3_snpswithin10bp_slop50|GRCh38_HG002_GIABv3.2.2_compoundhet_slop50|GRCh38_HG002_GIABv3.2.2_varswithin50bp|GRCh38_HG002_GIABv3.3.2_comphetindel10bp_slop50|GRCh38_HG002_GIABv3.3.2_comphetsnp10bp_slop50|GRCh38_HG002_GIABv3.3.2_complexandSVs|GRCh38_HG002_GIABv3.3.2_complexandSVs_alldifficultregions|GRCh38_HG002_GIABv3.3.2_complexindel10bp_slop50|GRCh38_HG002_GIABv3.3.2_notin_complexandSVs_alldifficultregions|GRCh38_HG002_GIABv3.3.2_snpswithin10bp_slop50|GRCh38_HG002_GIABv4.1_CNV_CCSandONT_elliptical_outlier|GRCh38_HG002_GIABv4.1_CNV_gt2assemblycontigs_ONTCanu_ONTFlye_CCSCanu|GRCh38_HG002_GIABv4.1_CNV_mrcanavarIllumina_CCShighcov_ONThighcov_intersection|GRCh38_HG002_GIABv4.1_CNVsandSVs|GRCh38_HG002_GIABv4.1_comphetindel10bp_slop50|GRCh38_HG002_GIABv4.1_comphetsnp10bp_slop50|GRCh38_HG002_GIABv4.1_complexandSVs|GRCh38_HG002_GIABv4.1_complexandSVs_alldifficultregions|GRCh38_HG002_GIABv4.1_complexindel10bp_slop50|GRCh38_HG002_GIABv4.1_inversions_slop25percent|GRCh38_HG002_GIABv4.1_notin_complexandSVs_alldifficultregions|GRCh38_HG002_GIABv4.1_othercomplexwithin10bp_slop50|GRCh38_HG002_GIABv4.1_snpswithin10bp_slop50|GRCh38_HG002_HG003_HG004_allsvs|GRCh38_HG002_Tier1plusTier2_v0.6.1|GRCh38_HG002_expanded_150__Tier1plusTier2_v0.6.1|GRCh38_HG003_GIABv3.3.2_comphetindel10bp_slop50|GRCh38_HG003_GIABv3.3.2_comphetsnp10bp_slop50|GRCh38_HG003_GIABv3.3.2_complexandSVs|GRCh38_HG003_GIABv3.3.2_complexandSVs_alldifficultregions|GRCh38_HG003_GIABv3.3.2_complexindel10bp_slop50|GRCh38_HG003_GIABv3.3.2_notin_complexandSVs_alldifficultregions|GRCh38_HG003_GIABv3.3.2_snpswithin10bp_slop50|GRCh38_HG004_GIABv3.3.2_comphetindel10bp_slop50|GRCh38_HG004_GIABv3.3.2_comphetsnp10bp_slop50|GRCh38_HG004_GIABv3.3.2_complexandSVs|GRCh38_HG004_GIABv3.3.2_complexandSVs_alldifficultregions|GRCh38_HG004_GIABv3.3.2_complexindel10bp_slop50|GRCh38_HG004_GIABv3.3.2_notin_complexandSVs_alldifficultregions|GRCh38_HG004_GIABv3.3.2_snpswithin10bp_slop50|GRCh38_HG005_GIABv3.3.2_comphetindel10bp_slop50|GRCh38_HG005_GIABv3.3.2_comphetsnp10bp_slop50|GRCh38_HG005_GIABv3.3.2_complexandSVs|GRCh38_HG005_GIABv3.3.2_complexandSVs_alldifficultregions|GRCh38_HG005_GIABv3.3.2_complexindel10bp_slop50|GRCh38_HG005_GIABv3.3.2_notin_complexandSVs_alldifficultregions|GRCh38_HG005_GIABv3.3.2_snpswithin10bp_slop50|GRCh38_HG005_HG006_HG007_MetaSV_allsvs|GRCh38_L1H_gt500|GRCh38_SimpleRepeat_diTR_11to50_slop5|GRCh38_SimpleRepeat_diTR_51to200_slop5|GRCh38_SimpleRepeat_diTR_gt200_slop5|GRCh38_SimpleRepeat_homopolymer_4to6_slop5|GRCh38_SimpleRepeat_homopolymer_7to11_slop5|GRCh38_SimpleRepeat_homopolymer_gt11_slop5|GRCh38_SimpleRepeat_imperfecthomopolgt10_slop5|GRCh38_SimpleRepeat_quadTR_20to50_slop5|GRCh38_SimpleRepeat_quadTR_51to200_slop5|GRCh38_SimpleRepeat_quadTR_gt200_slop5|GRCh38_SimpleRepeat_triTR_15to50_slop5|GRCh38_SimpleRepeat_triTR_51to200_slop5|GRCh38_SimpleRepeat_triTR_gt200_slop5|GRCh38_VDJ|GRCh38_alllowmapandsegdupregions|GRCh38_chainSelf|GRCh38_chainSelf_gt10kb|GRCh38_contigs_lt500kb|GRCh38_gaps_slop15kb|GRCh38_gc15_slop50|GRCh38_gc15to20_slop50|GRCh38_gc20to25_slop50|GRCh38_gc25to30_slop50|GRCh38_gc30to55_slop50|GRCh38_gc55to60_slop50|GRCh38_gc60to65_slop50|GRCh38_gc65to70_slop50|GRCh38_gc70to75_slop50|GRCh38_gc75to80_slop50|GRCh38_gc80to85_slop50|GRCh38_gc85_slop50|GRCh38_gclt25orgt65_slop50|GRCh38_gclt30orgt55_slop50|GRCh38_gt5segdups_gt10kb_gt99percidentity|GRCh38_lowmappabilityall|GRCh38_nonunique_l100_m2_e1|GRCh38_nonunique_l250_m0_e0|GRCh38_notinAllHomopolymers_gt6bp_imperfectgt10bp_slop5|GRCh38_notinAllTandemRepeatsandHomopolymers_slop5|GRCh38_notinalllowmapandsegdupregions|GRCh38_notinchainSelf|GRCh38_notinchainSelf_gt10kb|GRCh38_notinlowmappabilityall|GRCh38_notinsegdups|GRCh38_notinsegdups_gt10kb|GRCh38_segdups|GRCh38_segdups_gt10kb|TS_boundary|TS_contained' -Pv |cut -f 1,3,4,8,9,11|less -S > /tmp/$name$new_suffix		
	fi
done

grep '.*' -P  /tmp/*${new_suffix}  > /tmp/tmp.csv

cat /tmp/tmp.csv | awk -F ':' '{print $1"\t"$2}' > /tmp/table.csv

sed -i '1 i\\ttype\tregion\tFilter\trecal\tprecision\tF1_score' /tmp/table.csv


cat /tmp/table.csv | perl $perl_p/ff.PL 0 1 2 3 4 5 6 7| sed 's/ /,/g' | sed 's/.bed.gz//' > /tmp/table_.csv

mv /tmp/table_.csv /tmp/table.csv 
#ls table.csv 
cat /tmp/table.csv  | column -s, -t

echo ""
tfr t /tmp/table.csv  |& tee  /tmp/tmp.txt

echo -e "\n\n---now run on windows---\n\ttfr f `cat /tmp/tmp.txt` && ecd table.csv "
echo "" 


#rm /tmp/tmp.csv
rm /tmp/*${new_suffix}



