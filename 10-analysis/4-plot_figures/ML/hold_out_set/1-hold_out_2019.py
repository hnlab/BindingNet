import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem
from rdkit import DataStructs

from matplotlib import pyplot as plt
import seaborn as sns

if not Path('index_rm_all_simi_1').exists():
    Path('index_rm_all_simi_1').mkdir()

# PDBbind_minimized_intersected_Uw_df = pd.read_csv('/pubhome/xli02/project/PLIM/v2019_dataset/index/For_ML/PIP.csv', sep='\t')
PDBbind_v18 = pd.read_csv('INDEX_general_PL_data.2018.grepped', sep=' ', names=['pdb_id', '-logAffi'])

# PDBbind_general
# PDBbind_v19 = pd.read_csv('/pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019/INDEX_general_PL_data_grepped.2019', sep='\t')
# PDBbind_hold_out_2019 = PDBbind_v19[~PDBbind_v19['pdb_id'].isin(PDBbind_v18['pdb_id'])] # 1542
# pdbbind_v19_year = pd.read_csv('/pubhome/xli02/project/PLIM/analysis/20220829_paper/distribution/test.csv', sep=' ', header=None, names=['pdb_id', 'year'])
# not_2018 = PDBbind_hold_out_2019[~PDBbind_hold_out_2019['pdb_id'].isin(pdbbind_v19_year[pdbbind_v19_year['year'] > 2017]['pdb_id'])] # 104

# PDBbind_minimized
PDBbind_minimized = pd.read_csv('../../../3-property_calculate/2-other_property/PDBbind_dealt.csv', sep='\t')
PDBBind_v18_minimized = PDBbind_minimized[PDBbind_minimized['pdb_id'].isin(PDBbind_v18['pdb_id'])]
# PDBBind_v18_minimized.to_csv('minimized/PDBbind_v18_minimized.csv', sep='\t', index=False)
# len(PDBBind_v18_minimized) # 15666

PDBbind_hold_out_2019_minimized = PDBbind_minimized[~PDBbind_minimized['pdb_id'].isin(PDBbind_v18['pdb_id'])] # 1513


# PLANet_v18
index_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PLANet_Uw_dealt_df = pd.read_csv(f'{index_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv', sep='\t')
PLANet_v18 = PLANet_Uw_dealt_df[PLANet_Uw_dealt_df['Cry_lig_name'].isin(PDBBind_v18_minimized['pdb_id'])]
# PLANet_v18[['unique_identify', '-logAffi']].to_csv('minimized/PLANet_v18.csv', sep='\t', index=False)
# len(PLANet_v18) # 63933

# PLANet_hold_out_2019
PLANet_hold_out_2019 = PLANet_Uw_dealt_df[~PLANet_Uw_dealt_df['Cry_lig_name'].isin(PDBBind_v18_minimized['pdb_id'])]
# PLANet_hold_out_2019[['unique_identify', '-logAffi']].to_csv('index_rm_all_simi_1/PLANet_hold_out_2019.csv', sep='\t', index=False)
# len(PLANet_hold_out_2019) # 5883


# PDBbind_v18_subset
PDBbind_v18_subset = PDBBind_v18_minimized[PDBBind_v18_minimized['pdb_id'].isin(PLANet_v18['Cry_lig_name'])]
# PDBbind_v18_subset.to_csv('minimized/PDBbind_subset/PDBbind_v18_subset.csv', sep='\t', index=False)  # 5422


# PDBbind_hold_out_2019_subset
PDBbind_hold_out_2019_subset = PDBbind_hold_out_2019_minimized[PDBbind_hold_out_2019_minimized['pdb_id'].isin(PLANet_hold_out_2019['Cry_lig_name'])]
PDBbind_hold_out_2019_subset.to_csv('index_rm_all_simi_1/PDBbind_hold_out_2019_subset.csv', sep='\t', index=False) # 485


# 2. similarity
pdbbind_dir = '/pubhome/xli02/Downloads/dataset/PDBbind'
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
    # out_dir = f'index/similarity/{set_name_2}_in_{set_name_1}' #
    # if not Path(out_dir).exists():
    #     Path(out_dir).mkdir()
    simi_1_df = pd.DataFrame(lst, columns=cols)
    # simi_1_df.to_csv(f'{out_dir}/simi_1.csv', sep='\t', index=False)
    return simi_1_df, max_simi_for_mols_1


PDBbind_v18_subset_mols, PDBbind_v18_subset_mol_fps = obtain_fps(PDBbind_v18_subset, 'PDBbind_v18_subset')
PDBbind_hold_out_2019_subset_mols, PDBbind_hold_out_2019_subset_mol_fps = obtain_fps(PDBbind_hold_out_2019_subset, 'PDBbind_hold_out_2019_subset')
PLANet_v18_mols, PLANet_v18_mol_fps = obtain_fps(PLANet_v18, 'PLANet_v18')
PLANet_hold_out_2019_mols, PLANet_hold_out_2019_mol_fps = obtain_fps(PLANet_hold_out_2019, 'PLANet_hold_out_2019')

# 2.1 with PDBbind_hold_out
PLANet_simi_1_with_PDBbind_hold_out, max_simi_for_PDBbind_hold_with_PLANet_v18 = calculate_simi(PDBbind_hold_out_2019_subset_mols, PDBbind_hold_out_2019_subset_mol_fps, PDBbind_hold_out_2019_subset, PLANet_v18_mols, PLANet_v18_mol_fps, PLANet_v18, 'PDBbind_hold_out_2019_subset', 'PLANet_v18')
PDBbind_v18_simi_1_with_PDBbind_hold_out, max_simi_for_PDBbind_hold_with_PDBbind_v18 = calculate_simi(PDBbind_hold_out_2019_subset_mols, PDBbind_hold_out_2019_subset_mol_fps, PDBbind_hold_out_2019_subset, PDBbind_v18_subset_mols, PDBbind_v18_subset_mol_fps, PDBbind_v18_subset, 'PDBbind_hold_out_2019_subset', 'PDBbind_v18_subset')
# sns.kdeplot(max_simi_for_PDBbind_hold_with_PDBbind_v18)
# sns.kdeplot(max_simi_for_PDBbind_hold_with_PLANet_v18)
# plt.xlabel("Tanimoto similarity")
# plt.xlim(0,1)
# plt.title(f'Best similarity distribution among PDBbind_hold_out set and whole datasets')
# plt.legend(labels=['PDBbind_v18_subset','PLANet_v18'], title = "training_set")
# plt.savefig(f'/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019/index/similarity/PDBbind_hold_out_with_whole_set.png', dpi=300, bbox_inches='tight')
# plt.close()

# 2.2 with PLANet_hold_out
PLANet_v18_simi_1_with_PLANet_hold_out, max_simi_for_PLANet_hold_with_PLANet_v18 = calculate_simi(PLANet_hold_out_2019_mols, PLANet_hold_out_2019_mol_fps, PLANet_hold_out_2019, PLANet_v18_mols, PLANet_v18_mol_fps, PLANet_v18, 'PLANet_hold_out_2019', 'PLANet_v18')

PDBbind_v18_simi_1_with_PLANet_hold_out, max_simi_for_PLANet_hold_with_PDBbind_v18 = calculate_simi(PLANet_hold_out_2019_mols, PLANet_hold_out_2019_mol_fps, PLANet_hold_out_2019, PDBbind_v18_subset_mols, PDBbind_v18_subset_mol_fps, PDBbind_v18_subset, 'PLANet_hold_out_2019', 'PDBbind_v18_subset')

# sns.kdeplot(max_simi_for_PLANet_hold_with_PDBbind_v18)
# sns.kdeplot(max_simi_for_PLANet_hold_with_PLANet_v18)
# plt.xlabel("Tanimoto similarity")
# plt.xlim(0,1)
# plt.title(f'Best similarity distribution among PDBbind_hold_out set and whole datasets')
# plt.legend(labels=['PDBbind_v18_subset','PLANet_v18'], title = "training_set")
# plt.savefig(f'/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019/index/similarity/PLANet_hold_out_with_whole_set.png', dpi=300, bbox_inches='tight')
# plt.close()

# PDBbind_subset_rm_simi_1
# PDBbind_v18_simi_1_with_PDBbind_hold_out = pd.read_csv('index/similarity/PDBbind_v18_subset_in_PDBbind_hold_out_2019_subset/simi_1.csv', sep='\t')
# PDBbind_v18_simi_1_with_PLANet_hold_out = pd.read_csv('index/similarity/PDBbind_v18_subset_in_PLANet_hold_out_2019/simi_1.csv', sep='\t')
PDBbind_v18_subset_rm_simi_1 = PDBbind_v18_subset[~PDBbind_v18_subset['pdb_id'].isin(list(PDBbind_v18_simi_1_with_PDBbind_hold_out['PDBbind_v18_subset_cpnd_name']) + list(PDBbind_v18_simi_1_with_PLANet_hold_out['PDBbind_v18_subset_cpnd_name']))]
PDBbind_v18_subset_rm_simi_1.to_csv('index_rm_all_simi_1/PDBbind_v18_subset_rm_simi_1.csv', sep='\t', index=False)
# len(PDBbind_v18_subset_rm_simi_1) # 5731

# PLANet_simi_1_with_PDBbind_hold_out = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019/index/similarity/PLANet_v18_in_PDBbind_hold_out_2019_subset/simi_1.csv', sep='\t')
# PLANet_v18_simi_1_with_PLANet_hold_out = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019/index/similarity/PLANet_v18_in_PLANet_hold_out_2019/simi_1.csv', sep='\t')
PLANet_v18_rm_simi_1 = PLANet_v18[~PLANet_v18['unique_identify'].isin(list(PLANet_simi_1_with_PDBbind_hold_out['PLANet_v18_cpnd_name']) + list(PLANet_v18_simi_1_with_PLANet_hold_out['PLANet_v18_cpnd_name']))]
PLANet_v18_rm_simi_1[['unique_identify', '-logAffi']].to_csv('index_rm_all_simi_1/PLANet_v18_rm_simi_1.csv', sep='\t', index=False)
# len(PLANet_v18_rm_simi_1) # 63604

import os
os.system('cat index_rm_all_simi_1/PLANet_v18_rm_simi_1.csv index_rm_all_simi_1/PDBbind_v18_subset_rm_simi_1.csv > index_rm_all_simi_1/PDBbind_v18_subset_union_PLANet_rm_simi_1.csv')
os.system("sed -i '/pdb_id/d' index_rm_all_simi_1/PDBbind_v18_subset_union_PLANet_rm_simi_1.csv")
