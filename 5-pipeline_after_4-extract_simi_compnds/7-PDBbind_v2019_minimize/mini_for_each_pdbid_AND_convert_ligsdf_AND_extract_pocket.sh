#!/bin/bash
conda activate chemtools
pdbid=$1
script_dir=/home/xli/Documents/projects/ChEMBL-scaffold

export plop_path=/home/soft/plop/6.0
export LD_LIBRARY_PATH=$plop_path:$LD_LIBRARY_PATH
PDBbind_path=/home/xli/dataset/PDBbind_v2019/general_structure_only/$pdbid

# 1. AddH and generate parameter for crystal_ligand
cp ${PDBbind_path}/${pdbid}_ligand.mol2 cry_lig.mol2
/home/soft/ZBH/unicon/unicon -i cry_lig.mol2 -o cry_lig_addh.mol2 -p single > unicon_for_crylig.log
if [ ! -s cry_lig_addh.mol2 ]
then
echo "UNICON ERROR: addh of crystal_ligand error, YOU CAN TRY I-interpret, exit."
rm cry_lig_addh.mol2
exit
fi
python $script_dir/pipeline/7-PLOP_rescore/2-modify_crylig_mol2.py cry_lig_addh.mol2 cry_lig_addh_dealt.mol2 #
sed -i s/"LIG"/"CRL"/g cry_lig_addh_dealt.mol2 #

/home/qwang02/Software/schrodinger2016-2/utilities/mol2convert -imol2 cry_lig_addh_dealt.mol2 -omae lig.mae
/home/qwang02/Software/schrodinger2016-2/utilities/hetgrp_ffgen 2005 lig.mae
if [ ! -s crl ]
then
echo "PARAMETER ERROR, exit."
rm cry_lig_addh.mol2 cry_lig_addh_dealt.mol2 lig.mae
exit
fi

# 2. Run plop
obabel cry_lig_addh_dealt.mol2 -Ocry_lig_converted.pdb
grep ^ATOM cry_lig_converted.pdb > cry_lig_grepped.pdb
sed -i s/"ATOM  "/"HETATM"/g cry_lig_grepped.pdb
sed -i s/"A   1"/"88888"/g cry_lig_grepped.pdb

cp ${PDBbind_path}/${pdbid}_protein.pdb rec.pdb
python $script_dir/pipeline/7-PLOP_rescore/3-find_ions.py rec.pdb cry_lig_grepped.pdb rec_modfied.pdb > find_ions.log
grep -v "           H" rec_modfied.pdb > rec_modfied_rm_H.pdb
cat rec_modfied_rm_H.pdb cry_lig_grepped.pdb > cry_com.pdb
echo $script_dir/pipeline/pipeline_2/8-PDBbind_v2019_minimize/opt_cry_com_both.con | $plop_path/plop #
if [ ! -s cry_com_opt.pdb ]
then
echo "RECEPTOR_H_OPT ERROR: maybe caused by lack of metal parameter, exit."
rm cry_lig.mol2 cry_lig_addh.mol2 cry_lig_addh_dealt.mol2 lig.mae cry_lig_converted.pdb cry_lig_grepped.pdb rec.pdb rec_modfied.pdb rec_modfied_rm_H.pdb cry_com_opt.pdb
exit
fi
echo OPTIMIZATION RECEPTOR WITH CRYLIG DONE. 

# 3. obtain cry_lig_opt_converted.sdf and rec_h_opt.pdb
grep -v CRL cry_com_opt.pdb > rec_h_opt.pdb
grep CRL cry_com_opt.pdb > cry_lig_opt.pdb
/home/soft/ZBH/unicon/unicon -i  cry_lig_opt.pdb -o cry_lig_opt_converted.sdf -p single > unicon_for_crylig_sdf.log #
if [ ! -s cry_lig_opt_converted.sdf ]
then
echo "UNICON ERROR: addh of crystal_ligand error, YOU CAN TRY I-interpret, exit."
rm cry_lig.mol2 cry_lig_addh.mol2 cry_lig_addh_dealt.mol2 lig.mae crl cry_lig_converted.pdb cry_lig_grepped.pdb rec.pdb rec_modfied.pdb rec_modfied_rm_H.pdb cry_com_opt.pdb cry_lig_opt_converted.sdf
exit
fi

# 4. extract pocket within 6A
cp $script_dir/pipeline/pipeline_2/8-PDBbind_v2019_minimize/extract_pocket_6A_AND_calculate_RMSD.com . #
chimera --nogui extract_pocket_6A_AND_calculate_RMSD.com &> extract_pocket_6A.log

rm cry_lig.mol2 cry_lig_addh.mol2 cry_lig_addh_dealt.mol2 lig.mae crl cry_lig_converted.pdb cry_lig_grepped.pdb rec.pdb rec_modfied.pdb rec_modfied_rm_H.pdb cry_com_opt.pdb cry_lig_opt.pdb extract_pocket_6A_AND_calculate_RMSD.com
echo ALL DONE.
