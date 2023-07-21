#!/bin/bash
#$ -S /bin/bash
#$ -N chaC4_UwRcid_shuff_train
#$ -q ampere
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_asign_charge/2-train/scripts/rm_core_ids/PLANet_Uw/complex_6A/log/Uw_median_Rm_core_ids_cmx_asign_chrage_0_4.log
#$ -j y
#$ -r y
#$ -R y
#$ -pe ampere 20
#$ -soft -l hostname=!"(k224.hn.org|k225.hn.org)"
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim

# source /usr/bin/startcuda.sh
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
export CUDA_VISIBLE_DEVICES=0
python test_asign_charge/2-train/train_reduce_valid.py \
    --train-data dataset/true_lig_alone/bak/v3_before_PLANet_corrected_rec_charge/remove_core_ids_after_median_20211123/PLANet_Uw/complex_6A/PLIM_train_true_rec.hdf \
    --val-data dataset/true_lig_alone/bak/v3_before_PLANet_corrected_rec_charge/remove_core_ids_after_median_20211123/PLANet_Uw/complex_6A/PLIM_valid_true_rec.hdf \
    --checkpoint-dir test_asign_charge/train_res/rm_core_ids/PLANet_Uw/complex_6A/4/ \
    --dataset-name PLIM \
    --rec true_rec \
    --shuffle \
    --epochs 500
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# source /usr/bin/end_cuda.sh
