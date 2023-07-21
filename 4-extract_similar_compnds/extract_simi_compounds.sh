#!/bin/bash
#run under simi_compounds_dataset directory

data_dir=/home/xli/git/BindingNet
SCRIPT_FILE=$data_dir/4-extract_similar_compnds/extract_simi_compounds.py

cd $data_dir/v2019_dataset
ls web_client_CHEMBL* -d|awk -F '_' '{print $3}'| \
    parallel "qsub_anywhere.py \
        -c 'conda activate chemtools; python ${SCRIPT_FILE} -t {}' \
        -j 'web_client_{}' \
        -q honda \
        -N 'exract_simi_compnds' \
        --qsub_now"
