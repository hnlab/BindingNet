#!/bin/bash
ls /home/xli/dataset/PDBbind_v2019/general_structure_only/* -d | \
    parallel -k -j 4 'obabel {}/*ligand.mol2 -O{}/{/}_ligand.smi'