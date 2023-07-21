#!/bin/bash
#rec_mol2 = f'{target_pdb_dir}/{compnd}/rec_addcharge_pocket_6A.mol2'
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
cd $HOME/project/PLIM/deep_learning/FAST/fast_plim
python 1-featurize/featurize_split_data_PLANet_true_lig_alone_diff_split.py \
    --dataset-name PLANet \
    --input /pubhome/xli02/project/PLIM/v2019_dataset \
    --output dataset/true_lig_alone/diff_split/whole_set/PLANet/complex_6A/ \
    --metadata /pubhome/xli02/project/PLIM/v2019_dataset/index/For_ML/PLANet_Uw_final_median.csv \
    --rec true_rec
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
# &> log/PLANet_Uw_median.log
