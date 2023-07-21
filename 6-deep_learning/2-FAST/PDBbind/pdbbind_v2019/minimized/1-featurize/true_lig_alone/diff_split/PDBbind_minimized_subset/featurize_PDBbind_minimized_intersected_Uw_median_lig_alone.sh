#!/bin/bash
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
cd $HOME/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized
python 1-featurize/featurize_PDBbind_minimized_true_lig_alone_diff_split.py \
    --dataset-name PDBbind_v2019_minimized \
    --input /pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019 \
    --output dataset/diff_split/PDBbind_minimized_intersected_Uw/lig_alone/ \
    # --rec true_rec
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# &> log/PDBbind_minimized_intersected_Uw_median_lig_alone.log
