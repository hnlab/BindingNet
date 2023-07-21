#!/bin/bash
#$ -S /bin/bash
#$ -q cuda
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim

conda activate fast
export CUDA_VISIBLE_DEVICES=0

model_path=$1
test_path=`echo $model_path |awk -F'train_results/' '{print $2}'|awk -F'/best_checkpoint' '{print $1}'` #
num=`echo $test_path|awk -F'/' '{print $7}'` #
[ -d test_results/$test_path ] || mkdir -p test_results/$test_path #

model_name=$2
valid_set_hdf=$3
train_set_hdf=$4
test_set_hdf=$5
core_set_hdf=$6
core_intersected_Uw_set_hdf=$7
core_set_hdf_original=$8
core_intersected_Uw_set_hdf_original=$9

python 3-test/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLANet \
    --output test_results/$test_path \
    --subset-name valid \
    --test-data $valid_set_hdf \
    --title "${model_name}_validation_$num"

python 3-test/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLANet \
    --output test_results/$test_path \
    --subset-name train \
    --test-data $train_set_hdf \
    --title "${model_name}_training_$num"

python 3-test/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLANet \
    --output test_results/$test_path \
    --subset-name test \
    --test-data $test_set_hdf \
    --title "${model_name}_testing_$num"

python 3-test/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_minimized \
    --output test_results/$test_path \
    --subset-name CASF_v16_minimized \
    --test-data $core_set_hdf \
    --title "${model_name}_Test_on_CASF_minimized_$num"

python 3-test/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_minimized \
    --output test_results/$test_path \
    --subset-name CASF_v16_intersected_Uw_minimized \
    --test-data $core_intersected_Uw_set_hdf \
    --title "${model_name}_Test_on_CASF_intersected_Uw_minimized_$num"

python 3-test/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output test_results/$test_path \
    --subset-name CASF_v16_original \
    --test-data $core_set_hdf_original \
    --title "${model_name}_Test_on_CASF_original_$num"

python 3-test/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_original \
    --output test_results/$test_path \
    --subset-name CASF_v16_intersected_Uw_original \
    --test-data $core_intersected_Uw_set_hdf_original \
    --title "${model_name}_Test_on_CASF_intersected_Uw_original_$num"