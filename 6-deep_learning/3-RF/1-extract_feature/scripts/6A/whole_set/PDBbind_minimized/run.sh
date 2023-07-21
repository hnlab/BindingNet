#!/bin/bash
mkdir index_files
cd index_files
split -l 10000 /pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019/index/PDBbind_v19_minimized_succeed_manually_modified_final.csv
sed -i '1d' xaa
for i in *
do
sed -i '1i pdb_id\t-logAffi' $i
mv $i PDBbind_minimized_${i}.csv
qsub -N Pm${i} -o 1-extract_feature/scripts/6A/whole_set/PDBbind_minimized/log/${i}.log -j y ../PDBbind_minimized_feature_rf.sh $i
done
