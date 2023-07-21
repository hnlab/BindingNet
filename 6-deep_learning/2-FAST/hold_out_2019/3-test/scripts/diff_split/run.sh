#!/bin/bash

conda activate fast
export CUDA_VISIBLE_DEVICES=0
cd /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019

pipeline(){
    printf "%s Start testing on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
    cd 3-test
    # test in PLANet hold_out_2019
    mkdir -p ../test_result/test_on_PLANet_hold_out_2019
    python predict_only_for_multi_model.py \
        --models model_all_cmx.csv \
        --preprocessing-type raw \
        --feature-type pybel \
        --dataset-name PLANet \
        --output ../test_result/test_on_PLANet_hold_out_2019/ \
        --subset-name test \
        --test-data ../../dataset/true_lig_alone/whole_set/PLANet_hold_out_2019/complex_6A/PLANet_test_true_rec.hdf \
        --test_data_title PLANet_hold_out_2019 \
        &> scripts/log/test_on_PLANet_hold_out_2019_cmx.log

    python predict_only_for_multi_model.py \
        --models model_all_lig_alone.csv \
        --preprocessing-type raw \
        --feature-type pybel \
        --dataset-name PLANet \
        --output ../test_result/test_on_PLANet_hold_out_2019/ \
        --subset-name test \
        --test-data ../../dataset/true_lig_alone/whole_set/PLANet_hold_out_2019/lig_alone/PLANet_test_false_rec.hdf \
        --test_data_title PLANet_hold_out_2019 \
        &> scripts/log/test_on_PLANet_hold_out_2019_lig_alone.log

    # test in PDBbind hold_out_2019
    mkdir -p ../test_result/test_on_PDBbind_hold_out_2019
    python predict_only_for_multi_model.py \
        --models model_all_cmx.csv \
        --preprocessing-type raw \
        --feature-type pybel \
        --dataset-name PDBbind_v2019_minimized \
        --output ../test_result/test_on_PDBbind_hold_out_2019/ \
        --subset-name test \
        --test-data ../../PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_hold_out_2019/complex_6A/PDBbind_v19_minimized_test_true_rec.hdf \
        --test_data_title PDBbind_hold_out_2019 \
        &> scripts/log/test_on_PDBbind_hold_out_2019_cmx.log

    python predict_only_for_multi_model.py \
        --models model_all_lig_alone.csv \
        --preprocessing-type raw \
        --feature-type pybel \
        --dataset-name PDBbind_v2019_minimized \
        --output ../test_result/test_on_PDBbind_hold_out_2019/ \
        --subset-name test \
        --test-data ../../PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_hold_out_2019/lig_alone/PDBbind_v19_minimized_test_false_rec.hdf \
        --test_data_title PDBbind_hold_out_2019 \
        &> scripts/log/test_on_PDBbind_hold_out_2019_lig_alone.log

    printf "%s End testing on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
}
time pipeline
