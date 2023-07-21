'''
Filter conformations after rescoring:
    remove "LIG_DELTA" <= -20(twist or broken) or "TOTAL_DELTA_FIX" >= 100(clash);
    retain only top 10 "TOTAL_DELTA_FIX" conformations for each compound;
Add element symbol at column 77-78 of "filtered" pdb file, for reading by rdkit
Calculate "core_RMSD"
Remove "core_RMSD" >= 2
Add affinity data, Similarity, SMILES
input: 
    output.test.sorted; 
    aligned_core_sample_num.csv; 
    top-cmxminlig.pdb
    web_client_{target_chembl_id}-activity.tsv
    namelist
output:
    top-cmxminlig_filtered_dealt.pdb
    energy_sorted_core_rmsd.csv
'''
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import math
import time
import os

from rdkit import Chem
from rdkit.Chem import AllChem


def split_top_pdb(candi_pdb_file):
    with open(candi_pdb_file, 'r') as f1:
        lines = f1.readlines()

    lig = ''
    name_idx_to_lig = {}
    for line in lines:
        if 'TER' in line:
            lig = lig + 'TER\n'
            name_idx_to_lig[name_idx] = lig
            lig = ''
        else:
            newline = line
            if "ATOM" in line:
                if line.split()[2][1].isdigit():
                    ele_symbol = line.split()[2][0]
                else:
                    ele_symbol = line.split()[2][0:2].lower()    #“CL”要小写，否则会被rdkit读为“C”
                newline = line[:-1] + f'           {ele_symbol} \n'   #Add element symbol
            lig = lig + newline
            if 'nm' in line:
                name_idx = line.split()[1].split('=')[1]
    return name_idx_to_lig


def obtain_cry_mol(pdb_id, logger):
    PDBbind_fixed_sdf_path = '/home/xli/dataset/PDBBind_v2019_general_fixed_sdf'
    cry_lig_fixed_sdf = f'{PDBbind_fixed_sdf_path}/{pdb_id}_ligand.fixed.sdf'
    pdbbind_dir = '/home/xli/dataset/PDBbind_v2019/general_structure_only'
    cry_lig_sdf = f'{pdbbind_dir}/{pdb_id}/{pdb_id}_ligand.sdf'
    cry_lig_mol2 = f'{pdbbind_dir}/{pdb_id}/{pdb_id}_ligand.mol2'
    if Path(cry_lig_fixed_sdf).exists() and Chem.SDMolSupplier(cry_lig_fixed_sdf)[0] is not None:
        cry_mol = Chem.SDMolSupplier(cry_lig_fixed_sdf)[0]
    elif Chem.SDMolSupplier(cry_lig_sdf)[0] is not None:
        cry_mol = Chem.SDMolSupplier(cry_lig_sdf)[0]
        logger.warning(f"[{time.ctime()}] There is no {pdb_id}_ligand.fixed.sdf " \
            f"OR it cannot be read by rdkit, use {pdb_id}_ligand.sdf.")
    elif Chem.MolFromMol2File(cry_lig_mol2) is not None:
        cry_mol = Chem.MolFromMol2File(cry_lig_mol2)
        logger.warning(f"[{time.ctime()}] {pdb_id}_ligand.sdf also cannot be read by rdkit, " \
            f"use {pdb_id}_ligand.mol2.")
    else:
        logger.warning(f"[{time.ctime()}] Both {pdb_id}_ligand(.fixed).sdf and " \
            f"{pdb_id}_ligand.mol2 cannot be read by rdkit! skipped.")
        return None
    return cry_mol


def calc_core_RMSD(calc_df, name_idx_to_lig, cry_mol, info_df, logger):
    calc_df.loc[:,'core_RMSD'] = np.nan
    calc_df.loc[:,'MolWt'] = np.nan
    for row in calc_df.itertuples():
        mcs_smarts = info_df['mcs_smarts'][0]
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)

        candi_smi = info_df['similar_compounds_smiles'][0]
        candi_smi_mol = Chem.MolFromSmiles(candi_smi)

        mol_from_pdbblock = Chem.MolFromPDBBlock(name_idx_to_lig[row.NAME])
        if mol_from_pdbblock is None:
            calc_df.loc[row.Index, 'core_RMSD'] = 10
            calc_df.loc[row.Index, 'MolWt'] = Chem.rdMolDescriptors.CalcExactMolWt(candi_smi_mol)
            logger.warning(f"The pdb file after rescoring cannot be read by rdkit, " \
                f"suggesting error of this conformation. " \
                f"Cannot calculate the 'core_RMSD' of {row.NAME}, " \
                f"set as 10 and will be filtered.")
            continue
        if '.' in Chem.MolToSmiles(mol_from_pdbblock):
            Chem.MolToPDBFile(mol_from_pdbblock, f'{row.NAME}_before.pdb')
            os.system(f'obabel {row.NAME}_before.pdb -O{row.NAME}_after.pdb')
            mol_from_pdbblock = Chem.MolFromPDBFile(f'{row.NAME}_after.pdb')
            os.system(f'rm {row.NAME}_before.pdb {row.NAME}_after.pdb')

        try:
            candi_mol_assigned = AllChem.AssignBondOrdersFromTemplate(candi_smi_mol, mol_from_pdbblock)
        except:
            calc_df.loc[row.Index, 'core_RMSD'] = 10
            calc_df.loc[row.Index, 'MolWt'] = Chem.rdMolDescriptors.CalcExactMolWt(candi_smi_mol)
            logger.warning(f"AssignBondOrders failed, may caused by the wrong SMILES " \
                f"or 'H' in pdb files or covalent bond. " \
                f"Cannot calculate the 'core_RMSD' of {row.NAME}, "\
                f"set as 10 and will be filtered.")
            continue
        else:
            cry_mol_match = cry_mol.GetSubstructMatch(mcs_mol)
            matchIdx = int(row.NAME.split('-')[2])
            candi_mol_match = candi_mol_assigned.GetSubstructMatches(mcs_mol)[matchIdx]
            delta2 = 0.0
            for cry_i,candi_i in zip(cry_mol_match,candi_mol_match):
                d = (cry_mol.GetConformer().GetAtomPosition(cry_i) - candi_mol_assigned.GetConformer().GetAtomPosition(candi_i)).LengthSq()
                delta2 += d
            core_RMSD = math.sqrt(delta2/len(cry_mol_match))
            calc_df.loc[row.Index, 'core_RMSD'] = core_RMSD
            calc_df.loc[row.Index, 'MolWt'] = Chem.rdMolDescriptors.CalcExactMolWt(candi_smi_mol)


def add_affi(ene_df, target_chembl_id, info_df, affinity_df, logger):
    final_df = pd.DataFrame(
        {
            'Target_chembl_id':target_chembl_id,
            'Cry_lig_name':info_df['cry_lig_name'][0],
            'Cry_lig_smiles':info_df['cry_lig_smiles'][0],
            'Cry_lig_an':info_df['cry_lig_an'][0],
            'Similar_compnd_name':info_df['similar_compounds_name'][0],
            'Similar_compnd_smiles':info_df['similar_compounds_smiles'][0],
            'Similar_compnd_an':info_df['similar_compounds_an'][0],
            'Similarity':info_df['similarity'][0],
            'Affinity':np.nan,
            'Activity_id':np.nan,
            'Core_num':info_df['core_num'][0],
            'Diff_an':info_df['diff_an'][0],
            'Part_fix':info_df['part_fix'][0],
            'Total_sampled_num':info_df['sample_num'][0],
            'Similar_compnd_conform':ene_df['NAME'],
            'Total_delta':ene_df['TOTAL_DELTA_FIX'],
            'Lig_delta':ene_df['LIG_DELTA'],
            'Core_RMSD':ene_df['core_RMSD'],
            'MolWt':ene_df['MolWt'],
            'MCS_smarts':info_df['mcs_smarts'][0],
        }
    )
    
    filter_units_affi_df = affinity_df.query('standard_units in ["nM", "ug.mL-1"]').copy()
    
    row = list(final_df.itertuples())[0]   #`ene_df`对一个compound来说一定只有一行
    affi_rows = filter_units_affi_df[filter_units_affi_df['molecule_chembl_id']==row.Similar_compnd_name]
    if len(affi_rows) == 0:
        logger.warning(f"{target_chembl_id}_{row.Similar_compnd_name} has NO affinity " \
            f"with 'nM' or 'ug.mL-1' unit, " \
            f"Let 'Affinity' = 'No data', 'Activity_id' = 000000.")
        final_df.loc[row.Index, 'Affinity'] = 'No data'
        final_df.loc[row.Index, 'Activity_id'] = 000000
        return final_df

    if 'Kd' in set(affi_rows['standard_type']):
        filtered_rows = affi_rows.query('standard_type == "Kd"')
    elif 'Ki' in set(affi_rows['standard_type']):
        filtered_rows = affi_rows.query('standard_type == "Ki"')
    elif 'IC50' in set(affi_rows['standard_type']):
        filtered_rows =  affi_rows.query('standard_type == "IC50"')
    else:
        filtered_rows = affi_rows.query('standard_type == "EC50"')
    
    if '=' in set(filtered_rows['standard_relation']):
        relation_filtered_rows = filtered_rows.query('standard_relation == "="').copy()
    else:
        logger.warning(f"{target_chembl_id}_{row.Similar_compnd_name} only has affinity " \
            f"data with relation '<', use it as '='.")
        relation_filtered_rows = filtered_rows.query('standard_relation == "<"').copy()
    
    for relation_row in relation_filtered_rows.itertuples():
        if relation_row.standard_units == 'ug.mL-1':
            relation_filtered_rows.loc[relation_row.Index, 'standard_value'] = relation_row.standard_value * row.MolWt * 1000
            relation_filtered_rows.loc[relation_row.Index, 'standard_units'] = 'nM'
    
    best_affi_row = relation_filtered_rows.sort_values(by='standard_value').head(1)
    if len(relation_filtered_rows) > 1:
        logger.warning(f"{target_chembl_id}_{row.Similar_compnd_name} has " \
            f"{len(relation_filtered_rows)} different " \
            f"{best_affi_row['standard_type'].values[0]} affinity data, " \
            f"choose the best affinity data among them.")
    
    affi = f"{best_affi_row['standard_type'].values[0]} " \
        f"{best_affi_row['standard_relation'].values[0]} " \
        f"{best_affi_row['standard_value'].values[0]} " \
        f"{best_affi_row['standard_units'].values[0]}"
    final_df.loc[row.Index, 'Affinity'] = affi
    final_df.loc[row.Index, 'Activity_id'] = best_affi_row['activity_id'].values[0]

    final_df['Activity_id'] = final_df['Activity_id'].astype(int)
    final_df['Similar_compnd_an'] = final_df['Similar_compnd_an'].astype(int)
    final_df['Core_num'] = final_df['Core_num'].astype(int)
    final_df['Diff_an'] = final_df['Diff_an'].astype(int)
    final_df['Total_sampled_num'] = final_df['Total_sampled_num'].astype(int)
    return final_df

def final(target_chembl_id, pdb_id, compound_id, target_dir, lig_delta, total_delta_fix, coreRMSD):
    logger = logging.getLogger(__name__)
    error_code = 0

    cwd = Path('.').resolve()
    pose_dir = cwd / f'rescore/{target_chembl_id}_{pdb_id}_{compound_id}_pose'
    energy_file = str(pose_dir/f'output.test.sorted')
    energy_df = pd.read_csv(energy_file, sep=' ')

    info_file = str(cwd / f"aligned_core_sample_num.csv")
    info_df = pd.read_csv(info_file, sep="\t",dtype={'cry_lig_name':str})

    affinity_file = str(target_dir/f"web_client_{target_chembl_id}-activity.tsv")
    affinity_df = pd.read_csv(affinity_file, sep="\t")

    candi_pdb_file = str(pose_dir/f'top-cmxminlig.pdb')
    name_idx_to_lig = split_top_pdb(candi_pdb_file)

    cry_mol = obtain_cry_mol(pdb_id, logger)

    if len(energy_df) == 0:
        error_code = 9
        logger.warning(f"[{time.ctime()}] Error type 9: After rescoring, 0 conformation left for {target_chembl_id}_{pdb_id}_{compound_id}.")
        return error_code
    total_fix_filtered_df = energy_df[energy_df['TOTAL_DELTA_FIX'] < total_delta_fix].copy()    #remove "TOTAL_DELTA_FIX" >= 10000
    total_fix_lig_delta_filtered_df = total_fix_filtered_df[total_fix_filtered_df['LIG_DELTA'] > lig_delta].copy()  #remove "LIG_DELTA" <= -20
    filtered_top_10_df = total_fix_lig_delta_filtered_df.head(10).copy()   #retain only top 10 "TOTAL_DELTA_FIX" conformations for each compound

    # Print information
    namelist = str(cwd / f'rescore/namelist')
    with open(namelist, 'r') as f2:
        lines_1 = f2.readlines()
    comformation_for_rescore = [line.split()[2] for line in lines_1]
    aligned_sdf = str(sorted(cwd.rglob("*aligned_sample*.sdf"))[0])
    with open(aligned_sdf, 'r') as f5:
        lines_2 = f5.readlines()
    aligned_conformation = [f'{line.split()[0].split("_")[0]}-{line.split()[0].split("_")[2]}-{line.split()[0].split("_")[4]}'
        for line in lines_2 if "CHEMBL" in line]
    logger.info(f"[{time.ctime()}] For {target_chembl_id}_{pdb_id}_{compound_id}:")
    logger.info(f"[{time.ctime()}] Before rescoring, removed {set(aligned_conformation).difference(set(comformation_for_rescore))}, " \
        f"{len(comformation_for_rescore)} / {len(aligned_conformation)} conformations are left.")
    logger.info(f"[{time.ctime()}] After rescoring, removed {set(comformation_for_rescore).difference(set(energy_df['NAME']))}, " \
        f"{len(energy_df['NAME'])} conformations are left.")
    logger.info(f"[{time.ctime()}] According to the cutoff of 'TOTAL_DELTA_FIX', " \
        f"removed {set(energy_df['NAME']).difference(set(total_fix_filtered_df['NAME']))}, "\
        f"{len(total_fix_filtered_df['NAME'])} conformations left.")
    logger.info(f"[{time.ctime()}] According to the cutoff of 'LIG_DELTA', " \
        f"removed {set(total_fix_filtered_df['NAME']).difference(set(total_fix_lig_delta_filtered_df['NAME']))}, " \
        f"{len(total_fix_lig_delta_filtered_df['NAME'])} conformations left.")

    # Calculate core_RMSD
    if len(filtered_top_10_df) == 0:
        error_code = 10
        logger.warning(f"[{time.ctime()}] Error type 10: After filtering by cutoff of 'LIG_DELTA' and " \
            f"'TOTAL_DELTA_FIX', 0 conformation left for {target_chembl_id}_{pdb_id}_{compound_id}.")
        return error_code
    calc_core_RMSD(filtered_top_10_df, name_idx_to_lig, cry_mol, info_df, logger)

    # Sort, filter by core_RMSD
    final_ene_df = filtered_top_10_df.sort_values(
        by=['core_RMSD', 'TOTAL_DELTA_FIX', 'LIG_DELTA'], 
        ascending=[True, True, False]
        ).head(1).copy() 
    #retain only minimal "core_RMSD" conformations for each compound; if "core_RMSD" equals, retain minimal 'TOTAL_DELTA_FIX'
    if float(final_ene_df['core_RMSD']) >= coreRMSD:
        error_code = 11
        logger.warning(f"[{time.ctime()}] Error type 11: After filtering by cutoff of 'core_RMSD', " \
            f"0 conformation left for {target_chembl_id}_{pdb_id}_{compound_id}.")
        return error_code

    # Add affinity data
    logger.info(f"[{time.ctime()}] {len(filtered_top_10_df[filtered_top_10_df['core_RMSD'] == 10])} " \
        f"conformations cannot calculate 'coreRMSD', set as '10' .")
    logger.info(f"[{time.ctime()}] Finally obtain best conformation with minimal 'core_RMSD' conformation.")
    final_df = add_affi(final_ene_df, target_chembl_id, info_df, affinity_df, logger)

    # Output
    output_info_file = str(cwd / f'{target_chembl_id}_{pdb_id}_{compound_id}_final.csv')
    final_df.to_csv(output_info_file, sep = "\t", index = False) 

    final_ene_file = str(cwd / f'{target_chembl_id}_{pdb_id}_{compound_id}_dlig_{lig_delta}_dtotal_{total_delta_fix}_CoreRMSD_{coreRMSD}_ene.csv')
    final_ene_df.to_csv(final_ene_file, sep = "\t", index = False)

    filtered_pdb_file = str(cwd / f'{target_chembl_id}_{pdb_id}_{compound_id}_dlig_{lig_delta}_dtotal_{total_delta_fix}_CoreRMSD_{coreRMSD}_final.pdb')
    with open(filtered_pdb_file, 'w', newline='') as f3:
        for row in final_df.itertuples():
            for line in name_idx_to_lig[row.Similar_compnd_conform].split('\n'):
                if 'total_delta_fix' in line:
                    f3.write(line + f'  core_RMSD = {row.Core_RMSD}  ' \
                            f'Affinity = {row.Affinity.replace(" ", "")}  ' \
                            f'Similarity = {row.Similarity}  ' \
                            f'Part_fix = {row.Part_fix}\n')
                elif 'TER' in line:
                    f3.write(line)
                else:
                    f3.write(line+'\n')
    return error_code
