#!/bin/bash

for i in {1..5};
do
model_path=train_result/diff_split/PDBbind_minimized_subset/lig_alone/$i/best_checkpoint.pth
model_name=PDBbind_minimized_subset_lig_alone
valid_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw/lig_alone/${i}${i}${i}/PDBbind_v19_minimized_valid_false_rec.hdf
train_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw/lig_alone/${i}${i}${i}/PDBbind_v19_minimized_train_false_rec.hdf
test_set_hdf=dataset/diff_split/PDBbind_minimized_intersected_Uw/lig_alone/${i}${i}${i}/PDBbind_v19_minimized_test_false_rec.hdf
qsub -N PIP${i}lig -o 2-train/true_lig_alone/test/diff_split/whole_set/PDBbind_minimized_subset/lig_alone/log/${i}.log -j y ../../self_test.sh $model_path $model_name $valid_set_hdf $train_set_hdf $test_set_hdf
done
