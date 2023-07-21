#!/bin/bash

for i in {1..5};
do
model_path=train_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/whole_set/PLANet/complex_6A/$i/best_checkpoint.pth
model_name=PLANet_cmx
valid_set_hdf=dataset/true_lig_alone/diff_split/whole_set/PLANet/complex_6A/${i}${i}${i}/PLANet_valid_true_rec.hdf
train_set_hdf=dataset/true_lig_alone/diff_split/whole_set/PLANet/complex_6A/${i}${i}${i}/PLANet_train_true_rec.hdf
test_set_hdf=dataset/true_lig_alone/diff_split/whole_set/PLANet/complex_6A/${i}${i}${i}/PLANet_test_true_rec.hdf
qsub -N U${i}cmx -o 3-test/scripts/true_lig_alone_modify_dists/diff_split/whole_set/PLANet/complex_6A/log/${i}.log -j y ../../self_test_PLANet.sh $model_path $model_name $valid_set_hdf $train_set_hdf $test_set_hdf
done
