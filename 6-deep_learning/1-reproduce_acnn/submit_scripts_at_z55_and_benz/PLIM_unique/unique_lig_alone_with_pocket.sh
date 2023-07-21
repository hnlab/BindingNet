#!/bin/bash
#$ -S /bin/bash
#$ -N lig_U_within_target_acnn
#$ -q benz
#$ -o /dev/null
#$ -j y
#$ -r y
#$ -R y
#$ -pe benz 32
#$ -notify
##$ -now y
##$ -l h="k212.hn.org"

conda activate acnn-cpu 
cd $HOME/project/PLIM/deep_learning/acnn_can_ai_do/acnn_plim
pipeline(){
    printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
    python -u ACNN.py \
        -component ligand \
        -result_dir result/unique_ligand_alone_random_pocket \
        -subset PLIM_unique \
        -load_binding_pocket
    printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
}
time pipeline &> log/uniq_lig_alone_with_pocket.log
