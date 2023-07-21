#!/bin/bash

for i in {1..5};
do
model_path=train_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/whole_set/PDBbind_minimized_intersected_Uw_union_Uw/lig_alone/$i/best_checkpoint.pth
model_name=PDBbind_subset_union_PLANet_lig_alone
valid_set_hdf=dataset/true_lig_alone/diff_split/whole_set/PDBbind_minimized_intersected_Uw_union_Uw/lig_alone/${i}${i}${i}/PLANet_valid_false_rec.hdf
train_set_hdf=dataset/true_lig_alone/diff_split/whole_set/PDBbind_minimized_intersected_Uw_union_Uw/lig_alone/${i}${i}${i}/PLANet_train_false_rec.hdf
test_set_hdf=dataset/true_lig_alone/diff_split/whole_set/PDBbind_minimized_intersected_Uw_union_Uw/lig_alone/${i}${i}${i}/PLANet_test_false_rec.hdf
qsub -N PIPUP${i}lig -o 3-test/scripts/true_lig_alone_modify_dists/diff_split/whole_set/PDBbind_minimized_intersected_Uw_union_Uw/lig_alone/log/${i}.log -j y ../../self_test_PLANet.sh $model_path $model_name $valid_set_hdf $train_set_hdf $test_set_hdf
done
