#!/bin/bash
# 1. PLIM_all_cpx
python from_hdf_to_csv.py \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --test-data /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/complex_6A/PLIM_test_true_rec.hdf \
    --output /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/complex_6A/test_after.csv

python from_hdf_to_csv.py \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --test-data /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/complex_6A/PLIM_train_true_rec.hdf \
    --output /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/complex_6A/train_after.csv

python from_hdf_to_csv.py \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --test-data /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/complex_6A/PLIM_valid_true_rec.hdf \
    --output /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/complex_6A/valid_after.csv

# 2. PLIM_all_lig_alone
python from_hdf_to_csv.py \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --test-data /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/lig_alone/PLIM_train_false_rec.hdf \
    --output /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/lig_alone/train_after.csv

python from_hdf_to_csv.py \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --test-data /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/lig_alone/PLIM_valid_false_rec.hdf \
    --output /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/lig_alone/valid_after.csv

python from_hdf_to_csv.py \
    --preprocessing-type raw \
    --feature-type pybel \
    --dataset-name PLIM \
    --test-data /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/lig_alone/PLIM_test_false_rec.hdf \
    --output /pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/PLIM_all/lig_alone/test_after.csv
