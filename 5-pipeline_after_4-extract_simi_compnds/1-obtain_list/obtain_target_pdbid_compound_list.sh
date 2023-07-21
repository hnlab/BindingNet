#!/bin/bash
script_dir=/home/xli/git/BindingNet
for target in `ls $script_dir/v2019_dataset/web_client_CHEMBL* -d|awk -F '_' '{print $4}'`
do
cd $script_dir/v2019_dataset/web_client_$target
for pdbid_compound in `cat simi_0.7_fp_basic.csv|sed '1d'|awk '{print $1"_"$3}'`
do
echo "${target} ${pdbid_compound}" >> $script_dir/5-pipeline_after_4-extract_simi_compnds/1-obtain_list/all_target_pdbid_compound.list
done
done