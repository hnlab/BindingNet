#!/bin/bash
#$ -S /bin/bash
#$ -N acnn_lig_nu
#$ -q honda
#$ -o /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/log/tail_80000.log
#$ -j y
#$ -r y
#$ -notify
#$ -now y
#$ -l h="n117.hn.org"

# export HOSTNAME=$(hostname); 
conda activate acnn; 
python /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/ACNN.py \
    -component ligand \
    -dataset_start_idx 80000 \
    -dataset_end_idx 162295 \
    -result_dir /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/result/tail_80000_ligand_alone_random \
    -reload