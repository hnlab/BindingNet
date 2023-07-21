#!/bin/bash
#$ -S /bin/bash
#$ -N top4w_acnn_lig_nu
#$ -q honda
#$ -o /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/log/top_40000.log
#$ -j y
#$ -r y
#$ -notify
#$ -now y
#$ -l h="n135.hn.org"

# export HOSTNAME=$(hostname); 
conda activate acnn; 
python /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/ACNN.py \
    -component ligand \
    -dataset_end_idx 40000 \
    -result_dir /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/result/top_40000_ligand_alone_random \
    -reload