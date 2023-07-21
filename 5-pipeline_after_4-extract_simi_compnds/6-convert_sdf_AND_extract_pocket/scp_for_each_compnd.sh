#!/bin/bash
uniq_iden=$1
wrkdir=/home/xli/Documents/projects/ChEMBL-scaffold
z55_dir=/pubhome/xli02/project/PLIM/v2019_dataset
list_dir=$wrkdir/pipeline/pipeline_2/5_6_7-convert_sdf_extract_pocket
target=`echo $uniq_iden|awk -F '_' '{print $1}'`
pdbid=`echo $uniq_iden|awk -F '_' '{print $2}'`
compnd=`echo $uniq_iden|awk -F '_' '{print $3}'`
target_pdb_dir=$wrkdir/v2019_dataset/web_client_${target}/${target}_${pdbid}
cd $target_pdb_dir/$compnd
if [ ! -s cpnd_sdf_rec_pocket/rec_addcharge_pocket_6A.mol2 ]
then
echo $uniq_iden >> $list_dir/pocket_mol2_error.list
else
scp cpnd_sdf_rec_pocket/rec_addcharge_pocket_6A.mol2 z55:$z55_dir/web_client_${target}/${target}_${pdbid}/$compnd
fi

if [ ! -s cpnd_sdf_rec_pocket/compound.sdf ]
then
echo $uniq_iden >> $list_dir/compnd_sdf_error.list
else
scp cpnd_sdf_rec_pocket/compound.sdf z55:$z55_dir/web_client_${target}/${target}_${pdbid}/$compnd
fi
