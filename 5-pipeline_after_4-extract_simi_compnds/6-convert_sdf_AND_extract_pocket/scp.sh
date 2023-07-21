#!/bin/bash
script_dir=/home/xli/Documents/projects/ChEMBL-scaffold
script=$script_dir/pipeline/pipeline_2/5_6_7-convert_sdf_extract_pocket/scp_for_each_compnd.sh
final_list=$script_dir/v2019_dataset/index/bak/4-manually_correct_v1/PLIM_dataset_v1_final_modified.csv
cat $final_list |awk '{print $1}'|grep -v 'unique_identify'|parallel -k --joblog $script_dir/pipeline/pipeline_2/5_6_7-convert_sdf_extract_pocket/scp_job.log 'bash '$script' {}'

# grep $'\t'1$'\t' scp_job.log|awk -F '\t' '{print $9}'|parallel -k --joblog scp_job_2.log '{}'
