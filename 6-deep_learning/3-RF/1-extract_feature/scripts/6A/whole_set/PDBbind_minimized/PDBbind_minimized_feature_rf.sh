#!/bin/bash
#$ -S /bin/bash
#$ -q ampere
#$ -r y
#$ -notify
#$ -soft -l hostname=!"(k224.hn.org|k225.hn.org)"
#$ -wd /pubhome/xli02/project/PLIM/deep_learning/RFscore

printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
conda activate fast
name=$1
python 1-extract_feature/extract_rf_feature_for_PDBbind_6A.py \
    -dn PDBbind_minimized_${name} \
    -i 1-extract_feature/scripts/6A/whole_set/PDBbind_minimized/index_files/PDBbind_minimized_${name}.csv \
    -s /pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019 \
    -o featured_data/whole_set/PDBbind_minimized \
    -c 6
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
