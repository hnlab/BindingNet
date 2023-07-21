#!/bin/bash
#$ -S /bin/bash
#$ -N IUwRI_1ChL_shuff_epoch_500_train
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_asign_charge/2-train/scripts/rm_core_ids/PDBbind_intersected_Uw/lig_alone/log/PDBbind_intersected_Uw_median_Rm_core_ids_lig_alone_asign_charge_0_1.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
export CUDA_VISIBLE_DEVICES=0
python test_asign_charge/2-train/train_reduce_valid.py \
    --train-data PDBbind/pdbbind_v2019/original/dataset/PDBbind_v19_original_intersected_PLANet_Uw_Rm_core_ids/lig_alone/PDBbind_v19_original_train_false_rec.hdf \
    --val-data PDBbind/pdbbind_v2019/original/dataset/PDBbind_v19_original_intersected_PLANet_Uw_Rm_core_ids/lig_alone/PDBbind_v19_original_valid_false_rec.hdf \
    --checkpoint-dir test_asign_charge/train_res/rm_core_ids/PDBbind_intersected_Uw/lig_alone/1/ \
    --dataset-name PDBbind_v2019_original \
    --shuffle \
    --epochs 500
    # --rec true_rec \
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
