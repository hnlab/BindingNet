#!/bin/bash
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
cd $HOME/project/PLIM/deep_learning/FAST/fast_plim
python 1-featurize/featurize_split_data_combine_true_lig_alone_diff_split.py \
    --dataset-name PLANet \
    --input /pubhome/xli02/project/PLIM/v2019_dataset \
    --output dataset/true_lig_alone/diff_split/whole_set/PDBbind_minimized_v18_subset_union_PLANet_v18/lig_alone/
    # --train_data dataset/true_lig_alone/whole_set/PDBbind_minimized_v18_subset_union_PLANet_v18/complex_6A/train.csv \
    # --valid_data dataset/true_lig_alone/whole_set/PDBbind_minimized_v18_subset_union_PLANet_v18/complex_6A/valid.csv \
    # --test_data dataset/true_lig_alone/whole_set/PDBbind_minimized_v18_subset_union_PLANet_v18/complex_6A/test.csv
    # --rec true_rec
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# &> log/PDBbind_minimized_v18_subset_rm_simi_1_union_PLANet_v18_lig_alone.log
