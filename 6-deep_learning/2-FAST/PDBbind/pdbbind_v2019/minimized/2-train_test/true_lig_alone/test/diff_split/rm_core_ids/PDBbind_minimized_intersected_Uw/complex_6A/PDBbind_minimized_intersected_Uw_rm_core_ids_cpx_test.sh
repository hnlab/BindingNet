#!/bin/bash

for i in {1..5};
do
model_path=train_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/$i/best_checkpoint.pth
model_name=PDBbind_minimized_intersected_Uw_rm_CASF_ids_cmx
valid_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/${i}${i}${i}/PDBbind_v19_minimized_valid_true_rec.hdf
train_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/${i}${i}${i}/PDBbind_v19_minimized_train_true_rec.hdf
test_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/${i}${i}${i}/PDBbind_v19_minimized_test_true_rec.hdf
core_set_hdf=dataset/core_set/complex_6A/PDBbind_v19_minimized_test_true_rec.hdf
core_intersected_Uw_set_hdf=dataset/core_intersected_Uw/complex_6A/PDBbind_v19_minimized_test_true_rec.hdf
core_set_hdf_original=../original/dataset/CASF_v16/complex_6A/PDBbind_v19_original_test_true_rec.hdf
core_intersected_Uw_set_hdf_original=../original/dataset/CASF_v16_intersected_Uw/complex_6A/PDBbind_v19_original_test_true_rec.hdf
qsub -N PmIURI${i}cmx -o 2-train/true_lig_alone/test/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw/complex_6A/log/${i}.log -j y ../../test_on_core.sh $model_path $model_name $valid_set_hdf $train_set_hdf $test_set_hdf $core_set_hdf $core_intersected_Uw_set_hdf $core_set_hdf_original $core_intersected_Uw_set_hdf_original
done
