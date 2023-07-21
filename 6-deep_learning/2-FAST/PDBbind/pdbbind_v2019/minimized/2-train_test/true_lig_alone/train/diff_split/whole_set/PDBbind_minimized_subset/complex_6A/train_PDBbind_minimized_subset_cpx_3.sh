#!/bin/bash
#$ -S /bin/bash
#$ -N PIP_cpx3_shuff_epoch_500_train
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/2-train/true_lig_alone/train/diff_split/whole_set/PDBbind_minimized_subset/complex_6A/log/PDBbind_minimized_subset_complex_3.log
#$ -j y
#$ -r y
#$ -pe cuda 4
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
export CUDA_VISIBLE_DEVICES=0
python 2-train/train_reduce_valid.py \
    --train-data dataset/diff_split/PDBbind_minimized_intersected_Uw/complex_6A/333/PDBbind_v19_minimized_train_true_rec.hdf \
    --val-data dataset/diff_split/PDBbind_minimized_intersected_Uw/complex_6A/333/PDBbind_v19_minimized_valid_true_rec.hdf \
    --checkpoint-dir train_result/diff_split/PDBbind_minimized_subset/complex_6A/3/ \
    --dataset-name PDBbind_v2019_minimized \
    --rec true_rec \
    --shuffle \
    --epochs 500
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
