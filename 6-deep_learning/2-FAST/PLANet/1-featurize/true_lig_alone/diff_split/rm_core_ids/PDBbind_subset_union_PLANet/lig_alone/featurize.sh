#!/bin/bash
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
cd $HOME/project/PLIM/deep_learning/FAST/fast_plim
python 1-featurize/featurize_split_data_combine_true_lig_alone_diff_split.py \
    --dataset-name PLANet \
    --input /pubhome/xli02/project/PLIM/v2019_dataset \
    --output dataset/true_lig_alone/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/lig_alone/
    # --train_data dataset/true_lig_alone/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/train.csv \
    # --valid_data dataset/true_lig_alone/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/valid.csv \
    # --test_data dataset/true_lig_alone/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/test.csv
    # --rec true_rec
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# &> log/PDBbind_minimized_intersected_Uw_union_Uw_Rm_core_ids_lig_alone.log
