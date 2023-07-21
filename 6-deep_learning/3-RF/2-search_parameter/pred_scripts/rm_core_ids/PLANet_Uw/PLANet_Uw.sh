#!/bin/bash
#$ -S /bin/bash
#$ -N Uw_pred
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/deep_learning/RFscore/2-search_parameter/pred_scripts/rm_core_ids/PLANet_Uw/log/Uw_Rm_core_ids.log
#$ -j y
#$ -r y
#$ -l gpu=1
#$ -notify
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/RFscore

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
for i in {1..5}
do
python 2-search_parameter/main.py \
    --model RF \
    --feature_version VR1 \
    --output_path /pubhome/xli02/project/PLIM/deep_learning/RFscore/test_res \
    --core_data /pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019/CASF_v16_index_dealt.csv \
    --core_intersected_Uw /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/2-core_intersected_Uw/core_intersected_Uw.csv \
    --tvt_data_dir /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/true_lig_alone/rm_core_ids/PLANet_Uw/complex_6A \
    --dataset_name PLANet_Uw_Rm_core \
    --rf_max_features 8 \
    --rf_n_estimator 500 \
    --rep_num $i
done
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
