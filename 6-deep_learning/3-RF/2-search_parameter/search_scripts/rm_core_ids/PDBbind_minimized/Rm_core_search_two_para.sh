#!/bin/bash
#$ -S /bin/bash
#$ -N Pm_search_para_2
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/2-search_parameter/scripts/rm_core_ids/log/PDBbind_minimized_Rm_core_ids.log
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
    --tvt_data_dir /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_rm_core_ids/complex_6A \
    --dataset_name PDBbind_minimized_Rm_core
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
