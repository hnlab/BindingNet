#!/bin/bash
#$ -S /bin/bash
#$ -N all_acnn_lig_nu
#$ -q cuda
#$ -o /dev/null
#$ -j y
#$ -r y
##$ -l gpu=1
#$ -pe cuda 4
#$ -notify
##$ -now y
#$ -l h="k214.hn.org"

conda activate acnn 
cd $HOME/project/PLIM/deep_learning/acnn_can_ai_do
pipeline(){
    printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
    python -u ACNN.py \
        -component ligand \
        -result_dir result/all_ligand_alone_random
    printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
}
time pipeline &> log/all.log
