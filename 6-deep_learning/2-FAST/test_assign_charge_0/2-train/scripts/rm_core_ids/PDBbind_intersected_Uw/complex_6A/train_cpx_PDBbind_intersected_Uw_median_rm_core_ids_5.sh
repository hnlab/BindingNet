#!/bin/bash
#$ -S /bin/bash
#$ -N IUwRI_5ChC_shuff_epoch_500_train
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_asign_charge/2-train/scripts/rm_core_ids/PDBbind_intersected_Uw/complex_6A/log/PDBbind_intersected_Uw_median_Rm_core_ids_complex_asign_charge_0_5.log
#$ -j y
#$ -r y
#$ -pe cuda 4
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
export CUDA_VISIBLE_DEVICES=0
python test_asign_charge/2-train/train_reduce_valid.py \
    --train-data PDBbind/pdbbind_v2019/original/dataset/PDBbind_v19_original_intersected_PLANet_Uw_Rm_core_ids/complex_6A/PDBbind_v19_original_train_true_rec.hdf \
    --val-data PDBbind/pdbbind_v2019/original/dataset/PDBbind_v19_original_intersected_PLANet_Uw_Rm_core_ids/complex_6A/PDBbind_v19_original_valid_true_rec.hdf \
    --checkpoint-dir test_asign_charge/train_res/rm_core_ids/PDBbind_intersected_Uw/complex_6A/5/ \
    --dataset-name PDBbind_v2019_original \
    --rec true_rec \
    --shuffle \
    --epochs 500
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)

