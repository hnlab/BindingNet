#!/bin/bash

for i in {1..5};
do
model_path=train_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/rm_core_ids/PLANet_Uw/lig_alone/$i/best_checkpoint.pth
model_name=PLANet_Uw_rm_CASF_ids_lig_alone
valid_set_hdf=dataset/true_lig_alone/diff_split/rm_core_ids/PLANet_Uw/lig_alone/${i}${i}${i}/PLANet_valid_false_rec.hdf
train_set_hdf=dataset/true_lig_alone/diff_split/rm_core_ids/PLANet_Uw/lig_alone/${i}${i}${i}/PLANet_train_false_rec.hdf
test_set_hdf=dataset/true_lig_alone/diff_split/rm_core_ids/PLANet_Uw/lig_alone/${i}${i}${i}/PLANet_test_false_rec.hdf
core_set_hdf=PDBbind/pdbbind_v2019/minimized/dataset/core_set/lig_alone/PDBbind_v19_minimized_test_false_rec.hdf
core_intersected_Uw_set_hdf=PDBbind/pdbbind_v2019/minimized/dataset/core_intersected_Uw/lig_alone/PDBbind_v19_minimized_test_false_rec.hdf
core_set_hdf_original=PDBbind/pdbbind_v2019/original/dataset/CASF_v16/lig_alone/PDBbind_v19_original_test_false_rec.hdf
core_intersected_Uw_set_hdf_original=PDBbind/pdbbind_v2019/original/dataset/CASF_v16_intersected_Uw/lig_alone/PDBbind_v19_original_test_false_rec.hdf
qsub -N UwRI${i}lig -o 3-test/scripts/true_lig_alone_modify_dists/diff_split/rm_core_ids/PLANet_Uw/lig_alone/log/${i}.log -j y ../../test_on_core_PLANet.sh $model_path $model_name $valid_set_hdf $train_set_hdf $test_set_hdf $core_set_hdf $core_intersected_Uw_set_hdf $core_set_hdf_original $core_intersected_Uw_set_hdf_original
done
