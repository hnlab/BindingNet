#!/bin/bash
target=$1
pdbid=$2
compound_id=$3
script_dir=$4

mkdir rescore
cd rescore

## 3.0 Check rec_h_opt.pdb/log
rec_dir=$script_dir/v2019_dataset/web_client_${target}/${target}_${pdbid}/rec_opt
if [ ! -s $rec_dir/rec_h_opt.pdb ]
then
echo "RECEPTOR_H_OPT ERROR, exit."  >> ../rescore.log
cd ..
rm -r rescore
exit
fi

## 3.1 AddH
/home/soft/ZBH/unicon/unicon -i ../conform_mindist_greater_1_ligand.sdf -o ligands_addh.mol2 -p single > unicon.log
if [ ! -s ligands_addh.mol2 ]
then
echo "UNICON ERROR: addh of "${compound_id}" error, YOU CAN TRY I-interpret, exit."  >> ../rescore.log
mv unicon.log ..
cd ..
rm -r rescore
exit
fi

## 3.2 modify mol2 format
deal_mol2_script=$script_dir/pipeline/7-PLOP_rescore/1-modify_candi_mol2_for_rescore.py
python $deal_mol2_script ligands_addh.mol2 ligands_addh_dealt.mol2
sed -i s/LIG/LIG1/g ligands_addh_dealt.mol2
rm ligands_addh.mol2
echo "ADDH AND DEAL CANDI_MOL2 DONE."  >> ../rescore.log

## 3.3 Generate small molecule parameters!
export plop_script=/home/xli/script/mol2_plop_rescore2/zhouyu_rescore
export plop_path=/home/soft/plop/6.0
export mol2_script=/home/xli/script/mol2_plop_rescore2
export LD_LIBRARY_PATH=$plop_path:$LD_LIBRARY_PATH

pose=${target}_${pdbid}_${compound_id}_pose
$mol2_script/mol_split.pl ligands_addh_dealt.mol2  200  $pose #ligands_addh_dealt.mol2 -> xxx.het
echo "SPLIT MOL2 DONE. " >> ../rescore.log

$mol2_script/para_gen_opls2005.pl $pose ligands_addh_dealt.mol2 honda . 200 #ligands_addh_dealt.mol2 -> xxx.mol2 -> parameter
conf_num=`cat namelist |wc -l`
echo "GENERATE PARAMETER DONE. " >> ../rescore.log

## 3.4 Obtain rec_h_opt.pdb and rec_h_opt.log
cd $pose
cp $rec_dir/rec_h_opt.{log,pdb} .

## 3.5 Run plop rescoring!
rm -r *tmp
$plop_script/rescore/rescore2009/super_submit_2009.csh  $pose opls 2005 honda
echo "RESCORE DONE. " >> ../../rescore.log

## 3.6 calculate delta energy
sed -i 's/^[ ]*//g' *.ene
python ${mol2_script}/calculate_energy.py #sorted by "total_delta_fix"(= com - rec - ligfixmin) #That is to say: lig_model = "fix"

## 3.7 extract minimized structure
head -n $[$conf_num + 1] output.test.sorted | sed "/Sort/d" > output.top.sorted
perl ${mol2_script}/extract_from_decoys.pl top $conf_num #Add important energy terms in "top-cmxminlig.pdb", so that we can see from Chimera
rm -r cmxmin/ decoys/
rm *.ene output.test output.top.sorted
echo "EXTRACT POSE DONE." >> ../../rescore.log
cd ../..
