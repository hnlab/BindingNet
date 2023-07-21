#!/bin/bash
#$ -S /bin/bash
#$ -N PmIURI_1lig_shuff_epoch_500_train
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/2-train/true_lig_alone/train/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw/lig_alone/log/PDBbind_minimized_intersected_Uw_Rm_core_ids_lig_alone_1.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
export CUDA_VISIBLE_DEVICES=0
python 2-train/train_reduce_valid.py \
    --train-data dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/lig_alone/111/PDBbind_v19_minimized_train_false_rec.hdf \
    --val-data dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/lig_alone/111/PDBbind_v19_minimized_valid_false_rec.hdf \
    --checkpoint-dir train_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/lig_alone/1/ \
    --dataset-name PDBbind_v2019_minimized \
    --shuffle \
    --epochs 500
    # --rec true_rec \
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
