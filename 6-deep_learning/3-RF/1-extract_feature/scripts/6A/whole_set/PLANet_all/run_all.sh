#!/bin/bash
mkdir index_files
cd index_files
split -l 10000 /pubhome/xli02/project/PLIM/v2019_dataset/index/PLANet_all_final_median.csv
sed -i '1d' xaa
for i in *
do
sed -i '1i unique_identify\t-logAffi' $i
mv $i PLANet_all_${i}.csv
qsub -N A${i} -o 1-extract_feature/scripts/6A/whole_set/PLANet_all/log/${i}.log -j y ../PLANet_all_feature_rf.sh $i
done
