#!/bin/bash
#$ -S /bin/bash
#$ -N U_4lig_shuff_train
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/2-train/true_lig_alone_modify_dists/shuffle_true/epoch_500/diff_split/whole_set/PLANet/lig_alone/log/Uw_lig_alone_4.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast; 
export CUDA_VISIBLE_DEVICES=0
python 2-train/train_reduce_valid.py \
    --train-data dataset/true_lig_alone/diff_split/whole_set/PLANet/lig_alone/444/PLANet_train_false_rec.hdf \
    --val-data dataset/true_lig_alone/diff_split/whole_set/PLANet/lig_alone/444/PLANet_valid_false_rec.hdf \
    --checkpoint-dir train_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/whole_set/PLANet/lig_alone/4/ \
    --dataset-name PLANet \
    --shuffle \
    --epochs 500
    # --rec true_rec \
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
