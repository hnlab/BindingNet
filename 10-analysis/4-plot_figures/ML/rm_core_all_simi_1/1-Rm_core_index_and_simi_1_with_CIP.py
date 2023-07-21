import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem
from rdkit import DataStructs

wrkdir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/1-remove_same_id_in_core_set'
pdbbind_dir = '/pubhome/xli02/Downloads/dataset/PDBbind'
core_pdbid_file = f'{pdbbind_dir}/CASF-2016/core_pdbid.csv'
with open(core_pdbid_file, 'r') as f:
    lines = f.readlines()
core_pdbids = [line.rstrip() for line in lines]
# len(core_pdbids) # 285

def obtain_fps(data_df, dataset_name):
    mols = []
    mol_fps = []
    skipped_mols = []
    for row in data_df.itertuples():
        if 'pdb_id' in data_df.columns.values:
            uniq_id = row.pdb_id
            lig_file = f'{pdbbind_dir}/PDBbind_v2019/general_structure_only/{uniq_id}/{uniq_id}_ligand.smi'
            if not Path(lig_file).exists():
                print(f'{uniq_id} of PDBbind not exsits, skipped.')
                skipped_mols.append(uniq_id)
                continue
            compnd_mol = Chem.SmilesMolSupplier(lig_file, delimiter='\t', titleLine=False)[0]
        else:
            uniq_id = row.unique_identify
            if '_' in uniq_id:
                compnd_smi = PLANet_Uw_dealt_df[PLANet_Uw_dealt_df['unique_identify']==uniq_id]['Similar_compnd_smiles'].values[0]
                compnd_mol = Chem.MolFromSmiles(compnd_smi)
            else:
                lig_file = f'{pdbbind_dir}/PDBbind_v2019/general_structure_only/{uniq_id}/{uniq_id}_ligand.smi'
                if not Path(lig_file).exists():
                    print(f'{uniq_id} of PDBbind not exsits, skipped.')
                    skipped_mols.append(uniq_id)
                    continue
                compnd_mol = Chem.SmilesMolSupplier(lig_file, delimiter='\t', titleLine=False)[0]
        if compnd_mol is None:
            print(f'For compounds in {dataset_name}, {uniq_id} cannot be read by rdkit, skipped.')
            skipped_mols.append(uniq_id)
            continue
        compnd_mol.SetProp('_Name', uniq_id)
        mols.append(compnd_mol)
        mol_fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(compnd_mol,2))
    return mols, mol_fps


def calculate_simi(mols_1, mol_fps_1, affi_df_1, mols_2, mol_fps_2, affi_df_2, set_name_1, set_name_2):
    max_simi_for_mols_1 = []
    cols = [f'{set_name_1}_cpnd_name', f'{set_name_1}_cpnd_affi', f'{set_name_1}_cpnd_smiles', f'{set_name_2}_cpnd_name', f'{set_name_2}_cpnd_affi', f'{set_name_2}_cpnd_smiles', 'similarity']
    lst = []
    for i, mol in enumerate(mols_1):
        simi_list = DataStructs.BulkTanimotoSimilarity(mol_fps_1[i], mol_fps_2)
        mol_name = mol.GetProp('_Name')
        for j,simi in enumerate(simi_list):
            if simi == 1:
                if 'pdb_id' in affi_df_1.columns.values:
                    mol_affi = affi_df_1[affi_df_1['pdb_id']==mol_name]['-logAffi'].values[0]
                else:
                    mol_affi = affi_df_1[affi_df_1['unique_identify']==mol_name]['-logAffi'].values[0]
                mol_smiles = Chem.MolToSmiles(mol)
                mol_2_name = mols_2[j].GetProp('_Name')
                if 'pdb_id' in affi_df_2.columns.values:
                    mol_2_affi = affi_df_2[affi_df_2['pdb_id']==mol_2_name]['-logAffi'].values[0]
                else:
                    mol_2_affi = affi_df_2[affi_df_2['unique_identify']==mol_2_name]['-logAffi'].values[0]
                mol_2_smiles = Chem.MolToSmiles(mols_2[j])
                lst.append([mol_name, mol_affi, mol_smiles, mol_2_name, mol_2_affi, mol_2_smiles, simi])
        max_simi = max(simi_list)
        max_simi_for_mols_1.append(max_simi)
    # out_dir = f'{wrkdir}/similarity_analysisi/core_intersected_Uw/{set_name_2}_in_{set_name_1}' #
    # if not Path(out_dir).exists():
    #     Path(out_dir).mkdir()
    simi_1_df = pd.DataFrame(lst, columns=cols)
    # simi_1_df.to_csv(f'{out_dir}/simi_1.csv', sep='\t', index=False)
    return simi_1_df, max_simi_for_mols_1

# 1. BindingNet with CASF_subset
if not Path('index').exists():
    Path('index').mkdir()

PLANet_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PLANet_Uw_dealt_f = f'{PLANet_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv'
PLANet_Uw_dealt_df = pd.read_csv(PLANet_Uw_dealt_f, sep='\t')
# len(PLANet_Uw_dealt_df) # 69816

PLANet_Uw_rm_core_ids_df = PLANet_Uw_dealt_df[~((PLANet_Uw_dealt_df['Cry_lig_name'].isin(core_pdbids)) & (PLANet_Uw_dealt_df['Similarity'] == 1))]
# len(PLANet_Uw_rm_core_ids_df) # 69746

# core_intersected_Uw
core_df = pd.read_csv('CASF_v16_index_dealt.csv', sep='\t')
core_intersected_Uw_df = core_df[core_df['pdb_id'].isin(set(PLANet_Uw_dealt_df['Cry_lig_name']))].copy()
core_intersected_Uw_df.to_csv('index/core_intersected_Uw.csv', sep='\t', index=False)
# core_intersected_Uw_f = f'/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/2-core_intersected_Uw/core_intersected_Uw.csv'
# core_intersected_Uw_df = pd.read_csv(core_intersected_Uw_f, sep='\t')
core_intersected_Uw_mols, core_intersected_Uw_mol_fps = obtain_fps(core_intersected_Uw_df, 'CASF_v16_intersected_Uw')
# len(core_intersected_Uw_mols) # 115

# calculate similarity
Uw_mols, Uw_mol_fps = obtain_fps(PLANet_Uw_rm_core_ids_df, f'PLANet_Uw_') # 4min
Uw_simi_1_CIP, max_simi_for_core_intersected_Uw_test_Uw_train = calculate_simi(core_intersected_Uw_mols, core_intersected_Uw_mol_fps, core_intersected_Uw_df, Uw_mols, Uw_mol_fps, PLANet_Uw_rm_core_ids_df, 'CASF_v16_intersected_Uw_test', 'PLANet')
# len(Uw_mols) # 69746

# Uw_simi_1_CIP = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/1-remove_same_id_in_core_set/similarity_analysisi/core_intersected_Uw/PLANet_in_CASF_v16_intersected_Uw_test/simi_1.csv', sep='\t')

PLANet_Uw_rm_simi_CORE_1_df = PLANet_Uw_rm_core_ids_df[~PLANet_Uw_rm_core_ids_df['unique_identify'].isin(Uw_simi_1_CIP['PLANet_cpnd_name'])]
PLANet_Uw_rm_simi_CORE_1_df_final = PLANet_Uw_rm_simi_CORE_1_df[['unique_identify', '-logAffi']]
# len(PLANet_Uw_rm_simi_CORE_1_df_final) # 69706
PLANet_Uw_rm_simi_CORE_1_df_final.to_csv(f'index/PLANet_Uw_remove_core_set_ids.csv', sep='\t', index=False)

# 2. PDBbind_subset with CASF_subset
PDBbind_minimized_intersected_Uw_df = pd.read_csv(f'{PLANet_dir}/For_ML/PDBbind_subset.csv', sep='\t')
# len(PDBbind_minimized_intersected_Uw_df) # 5908
PDBbind_v19_minimized_intersected_Uw_remove_core_set_df = PDBbind_minimized_intersected_Uw_df[~PDBbind_minimized_intersected_Uw_df['pdb_id'].isin(core_pdbids)]
# len(PDBbind_v19_minimized_intersected_Uw_remove_core_set_df) # 5793

# PDBbind_minimized_intersected_Uw
PDBbind_intersected_Uw_mols, PDBbind_intersected_Uw_mol_fps = obtain_fps(PDBbind_v19_minimized_intersected_Uw_remove_core_set_df, 'PDBbind_minimized_intersected_Uw')
PIP_simi_1_CIP, max_simi_for_core_intersected_Uw_test_intersected_Uw_train = calculate_simi(core_intersected_Uw_mols, core_intersected_Uw_mol_fps, core_intersected_Uw_df, PDBbind_intersected_Uw_mols, PDBbind_intersected_Uw_mol_fps, PDBbind_v19_minimized_intersected_Uw_remove_core_set_df, 'CASF_v16_intersected_Uw_test', 'PDBbind_minimized_intersected_Uw')
# len(PDBbind_intersected_Uw_mols) # 5768

# PIP_simi_1_CIP = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/1-remove_same_id_in_core_set/similarity_analysisi/core_intersected_Uw/PDBbind_minimized_intersected_Uw_in_CASF_v16_intersected_Uw_test/simi_1.csv', sep='\t')
# len(PIP_simi_1_CIP) # 28

PIP_rm_simi_CORE_1_df = PDBbind_v19_minimized_intersected_Uw_remove_core_set_df[~PDBbind_v19_minimized_intersected_Uw_remove_core_set_df['pdb_id'].isin(PIP_simi_1_CIP['PDBbind_minimized_intersected_Uw_cpnd_name'])]
# len(PIP_rm_simi_CORE_1_df) # 5765
PIP_rm_simi_CORE_1_df.to_csv(f'index/PDBbind_minimized_intersected_Uw_rm_core_ids.csv', sep='\t', index=False)

import os
os.system('cat index/PLANet_Uw_remove_core_set_ids.csv index/PDBbind_minimized_intersected_Uw_rm_core_ids.csv > index/PDBbind_minimized_intersected_Uw_union_Uw_rm_core_ids.csv')
os.system("sed -i '/pdb_id/d' index/PDBbind_minimized_intersected_Uw_union_Uw_rm_core_ids.csv")
