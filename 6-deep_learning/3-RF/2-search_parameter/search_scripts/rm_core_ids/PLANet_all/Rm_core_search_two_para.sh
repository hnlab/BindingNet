#!/bin/bash
#$ -S /bin/bash
#$ -N A_search_para_2
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/2-search_parameter/scripts/rm_core_ids/log/PLANet_all_Rm_core_ids.log
#$ -j y
#$ -r y
#$ -R y
#$ -pe cuda 4
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/RFscore

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
python 2-search_parameter/search_best_param.py \
    --model RF \
    --feature_version VR1 \
    --tvt_data_dir /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/true_lig_alone/rm_core_ids/PLANet_all/complex_6A \
    --dataset_name PLANet_all_Rm_core
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
