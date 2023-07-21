#!/bin/bash
script_dir=/home/xli/git/BindingNet
opt_script=$script_dir/5-pipeline_after_4-extract_simi_compnds/5-PDBbind_v2019_minimize/mini_for_each_pdbid_AND_convert_ligsdf_AND_extract_pocket.sh

grep -v "#" /home/xli/dataset/PDBbind_v2019/raw/plain-text-index/index/INDEX_general_PL_data.2019 |awk '{print $1}' | while read pdbid
do
    PDBbind_path=/home/xli/dataset/PDBbind_v2019/general_structure_only/$pdbid
    if [ ! -s ${PDBbind_path}/${pdbid}_ligand.mol2 ]
    then
        echo "There is no structure of "$pdbid", skipped."
    else
        mkdir $script_dir/v2019_dataset/PDBbind_v2019/$pdbid
        cd $script_dir/v2019_dataset/PDBbind_v2019/$pdbid
        qsub_anywhere.py -c 'bash -i '$opt_script' '$pdbid'' \
            -j . \
            -q honda \
            -N opt \
            --qsub_now
    fi
done
