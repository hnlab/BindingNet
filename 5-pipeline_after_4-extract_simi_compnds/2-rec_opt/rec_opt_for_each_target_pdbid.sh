#!/bin/bash
conda activate chemtools
target=$1
pdbid=$2
script_dir=/home/xli/git/BindingNet

export plop_path=/home/soft/plop/6.0
export LD_LIBRARY_PATH=$plop_path:$LD_LIBRARY_PATH
PDBbind_path=/home/xli/dataset/PDBbind_v2019/general_structure_only/$pdbid
### 3.4.1 AddH and generate parameter for crystal_ligand
cp ${PDBbind_path}/${pdbid}_ligand.mol2 cry_lig.mol2
/home/soft/ZBH/unicon/unicon -i  cry_lig.mol2 -o cry_lig_addh.mol2 -p single > unicon_for_crylig.log
if [ ! -s cry_lig_addh.mol2 ]
then
echo "UNICON ERROR: addh of crystal_ligand error, YOU CAN TRY I-interpret, exit."
rm cry_lig_addh.mol2
exit
fi
python $script_dir/5-pipeline_after_4-extract_simi_compnds/scripts_for_plop/2-modify_crylig_mol2.py cry_lig_addh.mol2 cry_lig_addh_dealt.mol2
sed -i s/"LIG"/"CRY"/g cry_lig_addh_dealt.mol2

/home/qwang02/Software/schrodinger2016-2/utilities/mol2convert -imol2 cry_lig_addh_dealt.mol2 -omae lig.mae
/home/qwang02/Software/schrodinger2016-2/utilities/hetgrp_ffgen 2005 lig.mae

### 3.4.2 Run opt_cry_com.con
obabel cry_lig_addh_dealt.mol2 -Ocry_lig_converted.pdb
grep ^ATOM cry_lig_converted.pdb > cry_lig_grepped.pdb
sed -i s/"ATOM  "/"HETATM"/g cry_lig_grepped.pdb

cp ${PDBbind_path}/${pdbid}_protein.pdb rec.pdb
python $script_dir/5-pipeline_after_4-extract_simi_compnds/scripts_for_plop/3-find_ions.py rec.pdb cry_lig_grepped.pdb rec_modfied.pdb > find_ions.log
grep -v "           H" rec_modfied.pdb > rec_modfied_rm_H.pdb
cat rec_modfied_rm_H.pdb cry_lig_grepped.pdb > cry_com.pdb
echo $script_dir/5-pipeline_after_4-extract_simi_compnds/scripts_for_plop/opt_cry_com_ions_yes.con | $plop_path/plop
if [ ! -s cry_com_opt.pdb ]
then
echo "RECEPTOR_H_OPT ERROR: maybe caused by lack of metal parameter, exit."
rm cry_lig.mol2 cry_lig_addh.mol2 lig.mae cry_lig_converted.pdb rec_modfied_rm_H.pdb cry_com_opt.pdb cry_lig_addh_dealt.mol2
exit
fi
echo OPTIMIZATION RECEPTOR WITH CRYLIG DONE. 

### 3.4.3 Calculate the energy of rec_h_opt.pdb: run calculate_rec_ene.con
grep -v CRY cry_com_opt.pdb > rec_h_opt.pdb
echo $script_dir/5-pipeline_after_4-extract_simi_compnds/scripts_for_plop/calculate_rec_ene.con | $plop_path/plop
rm cry* lig.mae rec_modfied.pdb rec.pdb rec_modfied_rm_H.pdb
echo CALCULATING RECEPTOR ENERGY DONE.
