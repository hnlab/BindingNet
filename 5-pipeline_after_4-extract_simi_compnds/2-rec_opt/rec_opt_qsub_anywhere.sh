#!/bin/bash
script_dir=/home/xli/git/BindingNet
rec_opt_script=$script_dir/5-pipeline_after_4-extract_simi_compnds/2-rec_opt/rec_opt_for_each_target_pdbid.sh
cat $script_dir/5-pipeline_after_4-extract_simi_compnds/1-obtain_list/all_target_pdbid.list | while read line
do
target=`echo $line | awk '{print $1}'`
pdbid=`echo $line | awk '{print $2}'`
target_pdbid_rec_opt_dir=$script_dir/v2019_dataset/web_client_$target/${target}_${pdbid}/rec_opt
mkdir -p $target_pdbid_rec_opt_dir
cd $target_pdbid_rec_opt_dir
qsub_anywhere.py -c 'bash -i '$rec_opt_script' '$target' '$pdbid'' \
    -j . \
    -q honda \
    -N ''$target'_'$pdbid'' \
    --qsub_now
done
