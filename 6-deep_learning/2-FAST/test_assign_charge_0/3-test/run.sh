#!/bin/bash
#$ -S /bin/bash
#$ -q cuda
#$ -r y
#$ -l gpu=1
#$ -notify

cd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_asign_charge/3-test/

model_path=$1
test_path=`echo $model_path |awk -F'train_res/' '{print $2}'|awk -F'model' '{print $1}'`
num=`echo $test_path|awk -F'/' '{print $4}'`
python test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output ../test_res/test_on_core_intersected_Uw/$test_path \
    --subset-name CASF_v16_intersected_Uw \
    --test-data ../../PDBbind/pdbbind_v2019/original/dataset/CASF_v16_intersected_Uw/complex_6A/PDBbind_v19_original_test_true_rec.hdf \
    --title "PLANet_Uw_Rm_CASF_ids_Test_on_CASF_intersected_Uw_$num"
