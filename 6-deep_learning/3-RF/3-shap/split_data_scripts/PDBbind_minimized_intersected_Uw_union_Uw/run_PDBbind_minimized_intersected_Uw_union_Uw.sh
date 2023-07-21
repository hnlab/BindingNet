#!/bin/bash
mkdir splited_training_files
cd splited_training_files
split -l 4000 /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/true_lig_alone/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/train.csv
sed -i '1d' xaa
for i in *
do
sed -i '1i unique_identify\t-logAffi' $i
mv $i PDBbind_minimized_intersected_Uw_union_Uw_Rm_core_train_${i}.csv
qsub -N PmIUUw${i} -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/split_data_scripts/PDBbind_minimized_intersected_Uw_union_Uw/log/${i}.log -j y ../PDBbind_minimized_intersected_Uw_union_Uw_shap_calc.sh $i
done
