#!/bin/bash
#$ -S /bin/bash
#$ -q cuda
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized

conda activate fast
export CUDA_VISIBLE_DEVICES=0

model_path=$1
test_path=`echo $model_path |awk -F'train_result/' '{print $2}'|awk -F'/best_checkpoint' '{print $1}'`
num=`echo $test_path|awk -F'/' '{print $4}'` #
[ -d test_result/$test_path ] || mkdir -p test_result/$test_path

model_name=$2
valid_set_hdf=$3
train_set_hdf=$4
test_set_hdf=$5

python 2-train/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_minimized \
    --output test_result/$test_path \
    --subset-name valid \
    --test-data $valid_set_hdf \
    --title "${model_name}_validation_$num"

python 2-train/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_minimized \
    --output test_result/$test_path \
    --subset-name train \
    --test-data $train_set_hdf \
    --title "${model_name}_training_$num"

python 2-train/test_density.py \
    --checkpoint $model_path \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PDBbind_v2019_minimized \
    --output test_result/$test_path \
    --subset-name test \
    --test-data $test_set_hdf \
    --title "${model_name}_testing_$num"
