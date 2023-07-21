#!/bin/bash
#$ -S /bin/bash
#$ -q honda
#$ -o /dev/null
#$ -j y
#$ -r y
#$ -notify
#$ -wd /home/xli/Documents/projects/ChEMBL-scaffold

uniq_iden=$1
target=`echo $uniq_iden|awk -F '_' '{print $1}'`
pdbid=`echo $uniq_iden|awk -F '_' '{print $2}'`
compnd=`echo $uniq_iden|awk -F '_' '{print $3}'`
script_dir=/home/xli/Documents/projects/ChEMBL-scaffold
target_pdb_dir=$script_dir/v2019_dataset/web_client_${target}/${target}_${pdbid}
lig_pdb=$target_pdb_dir/$compnd/${uniq_iden}_dlig_-20_dtotal_100_CoreRMSD_2.0_final.pdb
# sdf_dir=$target_pdb_dir/$compnd/sdf
# [ -d $sdf_dir ] && rm -r $sdf_dir

cpnd_sdf_addcharge_pocket_dir=$target_pdb_dir/$compnd/cpnd_sdf_rec_pocket
mkdir $cpnd_sdf_addcharge_pocket_dir

pipeline(){
    echo `hostname`
    cd $cpnd_sdf_addcharge_pocket_dir
    /home/soft/ZBH/unicon/unicon -i $lig_pdb -o compound.sdf -p single > convert_sdf.log
    echo -e "open ../../rec_opt/rec_h_opt.pdb \n addcharge \n open compound.sdf \n sel #1 z<6 \n write format mol2 selected 0 rec_addcharge_pocket_6A.mol2 \n stop" | chimera --nogui > addcharge_extract_pocket_6A.log
    cd ..
    # rm extract_pocket* pocket*
}
pipeline &> $cpnd_sdf_addcharge_pocket_dir/$JOB_ID.${target}_${pdbid}_${compnd}.log
