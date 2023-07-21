#!/bin/bash
wrkdir=/home/lixl/data/from_x254/ChEMBL-scaffold
all_list=${wrkdir}/v2019_dataset/PDBbind_v2019/index/PDBbind_v19_minimized_succeed_manually_modified_final.csv
new_dir=${wrkdir}/planet_database/from_chembl_client/PDBbind_minimized
mkdir -p $new_dir
for pdbid in `cat $all_list |awk -F'\t' '{print $1}'|grep -v 'pdb_id'`
do
echo $pdbid
pdb_dir=$new_dir/${pdbid}
if [ ! -d $pdb_dir ];then
    mkdir $pdb_dir
    cp ${wrkdir}/v2019_dataset/PDBbind_v2019/$pdbid/cry_lig_opt_converted.sdf $pdb_dir/
    cp ${wrkdir}/v2019_dataset/PDBbind_v2019/$pdbid/rec_h_opt.pdb $pdb_dir/
    cp ${wrkdir}/v2019_dataset/PDBbind_v2019/$pdbid/rec_addcharge_pocket_6.mol2 $pdb_dir/
fi
done
