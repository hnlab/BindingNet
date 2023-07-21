#!/bin/bash

for i in {1..5};
do
model_path=train_result/diff_split/PDBbind_minimized_v18_subset/complex_6A/$i/best_checkpoint.pth
model_name=PDBbind_minimized_v18_subset_cmx
valid_set_hdf=dataset/diff_split/PDBbind_minimized_v18_subset/complex_6A/${i}${i}${i}/PDBbind_v19_minimized_valid_true_rec.hdf
train_set_hdf=dataset/diff_split/PDBbind_minimized_v18_subset/complex_6A/${i}${i}${i}/PDBbind_v19_minimized_train_true_rec.hdf
test_set_hdf=dataset/diff_split/PDBbind_minimized_v18_subset/complex_6A/${i}${i}${i}/PDBbind_v19_minimized_test_true_rec.hdf
qsub -N P18${i}cmx -o 2-train/true_lig_alone/test/diff_split/whole_set/PDBbind_minimized_v18_subset/complex_6A/log/${i}.log -j y ../../self_test.sh $model_path $model_name $valid_set_hdf $train_set_hdf $test_set_hdf
done
