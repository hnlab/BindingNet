#!/bin/bash
#$ -S /bin/bash
#$ -N chaL2_UwRcid_shuff_train
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_asign_charge/2-train/scripts/rm_core_ids/PLANet_Uw/lig_alone/log/Uw_median_Rm_core_ids_lig_alone_asign_chrage_0_2.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast; 
export CUDA_VISIBLE_DEVICES=0
python test_asign_charge/2-train/train_reduce_valid.py \
    --train-data dataset/true_lig_alone/bak/v3_before_PLANet_corrected_rec_charge/remove_core_ids_after_median_20211123/PLANet_Uw/lig_alone/PLIM_train_false_rec.hdf \
    --val-data dataset/true_lig_alone/bak/v3_before_PLANet_corrected_rec_charge/remove_core_ids_after_median_20211123/PLANet_Uw/lig_alone/PLIM_valid_false_rec.hdf \
    --checkpoint-dir test_asign_charge/train_res/rm_core_ids/PLANet_Uw/lig_alone/2/ \
    --dataset-name PLIM \
    --shuffle \
    --epochs 500
    # --rec true_rec \
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
