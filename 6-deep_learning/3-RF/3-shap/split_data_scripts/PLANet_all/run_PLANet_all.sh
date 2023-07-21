#!/bin/bash
mkdir splited_training_files
cd splited_training_files
split -l 4000 /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/true_lig_alone/rm_core_ids/PLANet_all/complex_6A/train.csv
sed -i '1d' xaa
for i in *
do
sed -i '1i unique_identify\t-logAffi' $i
mv $i PLANet_all_Rm_core_train_${i}.csv
qsub -N A${i} -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/split_data_scripts/PLANet_all/log/${i}.log -j y ../PLANet_all_shap_calc.sh $i
done
