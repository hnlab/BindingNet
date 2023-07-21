#!/bin/bash
#$ -S /bin/bash
#$ -q cuda
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_asign_charge/3-test/

conda activate fast
export CUDA_VISIBLE_DEVICES=0

model_path=$1
test_path=`echo $model_path |awk -F'rm_core_ids/' '{print $2}'|awk -F'model' '{print $1}'`
num=`echo $test_path|awk -F'/' '{print $3}'`
echo $model_path

python test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --output ../test_res/rm_core_ids/$test_path \
    --subset-name valid \
    --test-data ../../dataset/true_lig_alone/bak/v3_before_PLANet_corrected_rec_charge/remove_core_ids_after_median_20211123/PLANet_Uw/lig_alone/PLIM_valid_false_rec.hdf \
    --title "PLANet_Uw_Remove_CASF_ids_validation_lig_alone_asign_charge_$num"

python test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --output ../test_res/rm_core_ids/$test_path \
    --subset-name train \
    --test-data ../../dataset/true_lig_alone/bak/v3_before_PLANet_corrected_rec_charge/remove_core_ids_after_median_20211123/PLANet_Uw/lig_alone/PLIM_train_false_rec.hdf \
    --title "PLANet_Uw_Remove_CASF_ids_training_lig_alone_asign_charge_$num"

python test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --output ../test_res/rm_core_ids/$test_path \
    --subset-name test \
    --test-data ../../dataset/true_lig_alone/bak/v3_before_PLANet_corrected_rec_charge/remove_core_ids_after_median_20211123/PLANet_Uw/lig_alone/PLIM_test_false_rec.hdf \
    --title "PLANet_Uw_Remove_CASF_ids_testing_lig_alone_asign_charge_$num"

python test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output ../test_res/rm_core_ids/$test_path \
    --subset-name CASF_v16 \
    --test-data ../../PDBbind/pdbbind_v2019/original/dataset/CASF_v16/lig_alone/PDBbind_v19_original_test_false_rec.hdf \
    --title "PLANet_Uw_Rm_CASF_ids_Test_on_CASF_lig_alone_asign_charge_$num"

python test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output ../test_res/rm_core_ids/$test_path \
    --subset-name CASF_v16_intersected_Uw \
    --test-data ../../PDBbind/pdbbind_v2019/original/dataset/CASF_v16_intersected_Uw/lig_alone/PDBbind_v19_original_test_false_rec.hdf \
    --title "PLANet_Uw_Rm_CASF_ids_Test_on_CASF_intersected_Uw_lig_alone_asign_charge_$num"
