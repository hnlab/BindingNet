#!/bin/bash
wrkdir=/home/lixl/data/from_x254/ChEMBL-scaffold
all_list=${wrkdir}/v2019_dataset/index/PLANet_all_final_median.csv
new_dir=${wrkdir}/planet_database/from_chembl_client
mkdir -p $new_dir
for unique_iden in `cat $all_list|awk -F'\t' '{print $1}'|grep -v 'unique_identify'`
do
echo $unique_iden
target=`echo $unique_iden |awk -F'_' '{print $1}'`
pdbid=`echo $unique_iden |awk -F'_' '{print $2}'`
compound_id=`echo $unique_iden |awk -F'_' '{print $3}'`
pdb_dir=$new_dir/${pdbid}
if [ ! -d $pdb_dir ];then
    mkdir $pdb_dir
fi
if [ ! -f $pdb_dir/rec_h_opt.pdb ];then
    cp ${wrkdir}/v2019_dataset/web_client_${target}/${target}_${pdbid}/rec_opt/rec_h_opt.pdb $pdb_dir
fi
target_dir=$pdb_dir/target_${target}
if [ ! -d $target_dir ];then
    mkdir $target_dir
fi
compound_dir=$target_dir/${compound_id}
if [ ! -d $compound_dir ];then
    mkdir $compound_dir
fi
cp ${wrkdir}/v2019_dataset/web_client_${target}/${target}_${pdbid}/${compound_id}/*_final.pdb $compound_dir/${pdbid}_${target}_${compound_id}.pdb
cp ${wrkdir}/v2019_dataset/web_client_${target}/${target}_${pdbid}/${compound_id}/cpnd_sdf_rec_pocket/rec_addcharge_pocket_6A.mol2 $compound_dir
if [ -s ${wrkdir}/v2019_dataset/web_client_${target}/${target}_${pdbid}/${compound_id}/cpnd_sdf_rec_pocket/compound.sdf ];then
    cp ${wrkdir}/v2019_dataset/web_client_${target}/${target}_${pdbid}/${compound_id}/cpnd_sdf_rec_pocket/compound.sdf $compound_dir/${pdbid}_${target}_${compound_id}.sdf
else
    echo ${unique_iden} unicon convert sdf of compound error;
fi
done
