#!/bin/bash
source /usr/bin/startcuda.sh
conda activate acnn 
cd $HOME/project/PLIM/deep_learning/acnn_can_ai_do
pipeline(){
    printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
    python -u ACNN.py \
        -component ligand \
        -result_dir result/unique_cross_target_ligand_alone_random_modified_patience_5 \
        -subset PLIM_unique_cross_target \
        -patience 5
        # -patience 10
    printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
}
time pipeline &> log/uniq_cross_target_lig_alone_modified_patience_5.log
source /usr/bin/end_cuda.sh
