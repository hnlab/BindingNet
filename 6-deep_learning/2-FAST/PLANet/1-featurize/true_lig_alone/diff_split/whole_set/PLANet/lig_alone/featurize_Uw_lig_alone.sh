#!/bin/bash
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
cd $HOME/project/PLIM/deep_learning/FAST/fast_plim
python 1-featurize/featurize_split_data_PLANet_true_lig_alone_diff_split.py \
    --dataset-name PLANet \
    --input /pubhome/xli02/project/PLIM/v2019_dataset \
    --output dataset/true_lig_alone/diff_split/whole_set/PLANet/lig_alone/
    # --rec true_rec
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# &> log/PLANet_lig_alone.log
