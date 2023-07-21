#!/bin/bash
#$ -S /bin/bash
#$ -N Pm_shape
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/scripts/log/PDBbind_minimized_Rm_core_ids.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/RFscore

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
python 3-shap/cal_shap_value.py \
    --test_set /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_rm_core_ids/complex_6A/train.csv \
    -ds PDBbind_minimized \
    -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/shap_res
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
