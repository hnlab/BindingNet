#!/bin/bash
#rec_mol2 = f'{args.input}/{name}/rec_addcharge_pocket_6.mol2'
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
cd $HOME/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized
python 1-featurize/featurize_PDBbind_minimized_true_lig_alone_diff_split.py \
    --dataset-name PDBbind_v2019_minimized \
    --input /pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019 \
    --output dataset/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/ \
    --metadata /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/1-remove_same_id_in_core_set/PDBbind_minimized_intersected_Uw/PDBbind_minimized_intersected_Uw_rm_core_ids.csv \
    --rec true_rec
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# &> log/PDBbind_minimized_intersected_Uw_median_rm_core_set_ids_cpx.log
