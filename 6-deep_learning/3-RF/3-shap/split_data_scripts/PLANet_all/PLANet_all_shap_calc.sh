#!/bin/bash
#$ -S /bin/bash
#$ -q ampere
#$ -r y
#$ -notify
#$ -soft -l hostname=!"(k224.hn.org|k225.hn.org)"
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/RFscore

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
name=$1
python 3-shap/cal_shap_value_splited.py \
    -t 3-shap/split_data_scripts/PLANet_all/splited_training_files/PLANet_all_Rm_core_train_${name}.csv \
    -ds PLANet_all \
    -idx ${name} \
    -o 3-shap/split_shap_res/rm_core_ids
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
