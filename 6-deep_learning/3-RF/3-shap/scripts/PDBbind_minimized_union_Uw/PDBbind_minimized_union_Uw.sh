#!/bin/bash
#$ -S /bin/bash
#$ -N PmUUw_shape
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/scripts/log/PDBbind_minimized_union_Uw_Rm_core_ids.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/RFscore

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
python 3-shap/cal_shap_value.py \
    --test_set /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/true_lig_alone/rm_core_ids/PDBbind_minimized_union_Uw/complex_6A/train.csv \
    -ds PDBbind_minimized_union_Uw \
    -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/shap_res
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
