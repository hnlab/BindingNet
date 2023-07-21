#!/bin/bash
#$ -S /bin/bash
#$ -N PLIM
#$ -q honda
#$ -o /dev/null
#$ -j y
#$ -r y
#$ -notify
##$ -now y
#$ -t 1-195686

pipeline(){
    echo `hostname`
    python -m cProfile ${python_script} \
        -t ${target} \
        -p ${pdb_id} \
        -c ${compound_id} \
        -wd ${wrkdir} \
        -ODN ${output_dataset_name} \
        -scc ${serious_clash_cutoff} \
        -dlig ${lig_delta} \
        -dtotal ${total_delta_fix} \
        -CR ${coreRMSD}
}

conda activate chemtools
SEED=`tr -cd 0-9 </dev/urandom | head -c 8`
tmp_dir=/tmp/$USER/tmp.$SEED
mkdir -p $tmp_dir
cd $tmp_dir

wrkdir=/home/xli/git/BindingNet/
python_script=${wrkdir}/5-pipeline_after_4-extract_simi_compnds/3-align-filter_clash-rescore-final_filter/run_for_each_compound/run_for_each_compound.py
output_dataset_name=v2019_dataset
serious_clash_cutoff=1
lig_delta=-20
total_delta_fix=100
coreRMSD=2
list=${wrkdir}/5-pipeline_after_4-extract_simi_compnds/1-obtain_list/all_target_pdbid_compound.list
target=`sed -n ${SGE_TASK_ID}p ${list} | awk '{print $1}'`
pdb_id=`sed -n ${SGE_TASK_ID}p ${list} | awk '{print $2}'|awk -F '_' '{print $1}'`
compound_id=`sed -n ${SGE_TASK_ID}p ${list} | awk -F '_' '{print $2}'`

pipeline > $tmp_dir/$JOB_ID.$SGE_TASK_ID.${target}_${pdb_id}_${compound_id}.log
home_dir=${wrkdir}/${output_dataset_name}/web_client_${target}/${target}_${pdb_id}/${compound_id}
mkdir -p $home_dir
mv $tmp_dir/* $home_dir
cd ..
rmdir $tmp_dir
