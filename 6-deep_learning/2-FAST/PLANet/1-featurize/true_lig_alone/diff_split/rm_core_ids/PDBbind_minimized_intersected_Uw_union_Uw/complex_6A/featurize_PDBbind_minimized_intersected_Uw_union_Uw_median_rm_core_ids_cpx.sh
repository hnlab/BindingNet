#!/bin/bash
#rec_mol2 = f'{target_pdb_dir}/{compnd}/rec_addcharge_pocket_6A.mol2'
#rec_mol2 = f'{pdbbind_minimized_dir}/{name}/rec_addcharge_pocket_6.mol2'
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
cd $HOME/project/PLIM/deep_learning/FAST/fast_plim
python 1-featurize/featurize_split_data_combine_true_lig_alone_diff_split.py \
    --dataset-name PLANet \
    --input /pubhome/xli02/project/PLIM/v2019_dataset \
    --output dataset/true_lig_alone/diff_split/rm_core_ids/PDBbind_minimized_intersected_Uw_union_Uw/complex_6A/ \
    --metadata test_on_core_set/1-remove_same_id_in_core_set/PDBbind_minimized_intersected_Uw_union_Uw/PDBbind_minimized_intersected_Uw_union_Uw_rm_core_ids.csv \
    --rec true_rec
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# &> log/PDBbind_minimized_intersected_Uw_union_Uw_median_rm_core_ids.log
