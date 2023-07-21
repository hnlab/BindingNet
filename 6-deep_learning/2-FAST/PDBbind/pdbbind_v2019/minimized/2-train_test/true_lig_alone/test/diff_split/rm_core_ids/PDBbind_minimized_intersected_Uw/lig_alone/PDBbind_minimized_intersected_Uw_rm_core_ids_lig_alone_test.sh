#!/bin/bash

for i in {1..5};
do
model_path=train_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/lig_alone/$i/best_checkpoint.pth
model_name=PDBbind_minimized_intersected_Uw_rm_CASF_ids_lig_alone
valid_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/lig_alone/${i}${i}${i}/PDBbind_v19_minimized_valid_false_rec.hdf
train_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/lig_alone/${i}${i}${i}/PDBbind_v19_minimized_train_false_rec.hdf
test_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/lig_alone/${i}${i}${i}/PDBbind_v19_minimized_test_false_rec.hdf
core_set_hdf=dataset/core_set/lig_alone/PDBbind_v19_minimized_test_false_rec.hdf
core_intersected_Uw_set_hdf=dataset/core_intersected_Uw/lig_alone/PDBbind_v19_minimized_test_false_rec.hdf
core_set_hdf_original=../original/dataset/CASF_v16/lig_alone/PDBbind_v19_original_test_false_rec.hdf
core_intersected_Uw_set_hdf_original=../original/dataset/CASF_v16_intersected_Uw/lig_alone/PDBbind_v19_original_test_false_rec.hdf
qsub -N PmIURI${i}lig -o 2-train/true_lig_alone/test/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw/lig_alone/log/${i}.log -j y ../../test_on_core.sh $model_path $model_name $valid_set_hdf $train_set_hdf $test_set_hdf $core_set_hdf $core_intersected_Uw_set_hdf $core_set_hdf_original $core_intersected_Uw_set_hdf_original
done
