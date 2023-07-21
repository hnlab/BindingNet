#!/bin/bash

script_dir=/home/xli/Documents/projects/ChEMBL-scaffold
# final_list=$script_dir/v2019_dataset/index/PLIM_dataset_v1_final.csv
final_list=$script_dir/v2019_dataset/index/bak/4-manually_correct_v1/PLIM_dataset_v1_final_modified.csv
cat $final_list |awk '{print $1}'|grep -v 'unique_identify'|parallel -k 'qsub -N {} run_for_each_iden.sh {}'
