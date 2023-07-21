#!/bin/bash
#$ -S /bin/bash
#$ -q cuda
#$ -r y
#$ -notify
#$ -l ngpus=1
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/RFscore

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
name=$1
python 3-shap/cal_shap_value_splited.py \
    -t 3-shap/split_data_scripts/PDBbind_minimized/splited_training_files/PDBbind_minimized_Rm_core_train_${name}.csv \
    -ds PDBbind_minimized \
    -idx ${name} \
    -o 3-shap/split_shap_res/rm_core_ids
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
