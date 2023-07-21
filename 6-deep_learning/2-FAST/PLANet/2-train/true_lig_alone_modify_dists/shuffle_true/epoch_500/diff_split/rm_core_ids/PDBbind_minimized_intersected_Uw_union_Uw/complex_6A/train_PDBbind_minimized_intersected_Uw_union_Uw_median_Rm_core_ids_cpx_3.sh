#!/bin/bash
#$ -S /bin/bash
#$ -N PIPUPRI3cpx_shuff_train
#$ -q ampere
#$ -o /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/2-train/true_lig_alone_modify_dists/shuffle_true/epoch_500/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/log/PDBbind_minimized_intersected_Uw_union_Uw_Rm_core_ids_complex_6A_3.log
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
python 2-train/train_reduce_valid.py \
    --train-data dataset/true_lig_alone/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/333/PLANet_train_true_rec.hdf \
    --val-data dataset/true_lig_alone/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/333/PLANet_valid_true_rec.hdf \
    --checkpoint-dir train_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/3/ \
    --dataset-name PLANet \
    --rec true_rec \
    --shuffle \
    --epochs 500
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# source /usr/bin/end_cuda.sh
