#!/bin/bash
#$ -S /bin/bash
#$ -N cpx_U_cross_target_acnn
#$ -q benz
#$ -o /dev/null
#$ -j y
#$ -r y
#$ -R y
#$ -pe benz 32
#$ -notify
##$ -now y
##$ -l h="k228.hn.org"

conda activate acnn-cpu 
cd $HOME/project/PLIM/deep_learning/acnn_can_ai_do/acnn_plim
pipeline(){
    printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
    python -u ACNN.py \
        -component binding \
        -result_dir result/unique_cross_target_complex_random \
        -subset PLIM_unique_cross_target \
        -load_binding_pocket
        # -patience 5
        # -patience 10
    printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
}
time pipeline &> log/uniq_cross_target_complex_random.log
