import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem
from rdkit import DataStructs

pdbbind_dir = '/pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019'

def obtain_training_fps(data_df, dataset_name):
    mols = []
    mol_fps = []
    skipped_mols = []
    for row in data_df.itertuples():
        if 'pdb_id' in data_df.columns.values:
            uniq_id = row.pdb_id
            lig_file = f'{pdbbind_dir}/general_structure_only/{uniq_id}/{uniq_id}_ligand.smi'
            if not Path(lig_file).exists():
                print(f'{uniq_id} of PDBbind not exsits, skipped.')
                skipped_mols.append(uniq_id)
                continue
            compnd_mol = Chem.SmilesMolSupplier(lig_file, delimiter='\t', titleLine=False)[0]
        else:
            uniq_id = row.unique_identify
            if '_' in uniq_id:
                if uniq_id not in list(PLANet_smiles_df['unique_identify']):
                    print(f'{uniq_id} of PLANet not exsits in smiles_df, skipped.')
                    skipped_mols.append(uniq_id)
                    continue
                compnd_smi = PLANet_smiles_df[PLANet_smiles_df['unique_identify']==uniq_id]['Similar_compnd_smiles'].values[0]
                compnd_mol = Chem.MolFromSmiles(compnd_smi)
            else:
                lig_file = f'{pdbbind_dir}/general_structure_only/{uniq_id}/{uniq_id}_ligand.smi'
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

def calculate_simi(hold_2019_mols, hold_2019_mol_fps, hold_2019_df, mols_2, mol_fps_2, affi_df_2, set_name_1, set_name_2):
    max_simi_for_hold_2019_mols = []
    cols = [f'{set_name_1}_name', f'{set_name_1}_affi', f'{set_name_1}_smiles', f'{set_name_2}_cpnd_name', f'{set_name_2}_cpnd_affi', f'{set_name_2}_cpnd_smiles', 'similarity']
    lst = []
    for i, mol in enumerate(hold_2019_mols):
        simi_list = DataStructs.BulkTanimotoSimilarity(hold_2019_mol_fps[i], mol_fps_2)
        mol_name = mol.GetProp('_Name')
        for j,simi in enumerate(simi_list):
            if simi > 0.8:
                # mol_affi = hold_2019_df[hold_2019_df['pdb_id']==mol_name]['-logAffi'].values[0]
                if 'pdb_id' in hold_2019_df.columns.values:
                    mol_affi = hold_2019_df[hold_2019_df['pdb_id']==mol_name]['-logAffi'].values[0]
                else:
                    mol_affi = hold_2019_df[hold_2019_df['unique_identify']==mol_name]['-logAffi'].values[0]
                mol_smiles = Chem.MolToSmiles(mol)
                mol_2_name = mols_2[j].GetProp('_Name')
                if 'pdb_id' in affi_df_2.columns.values:
                    mol_2_affi = affi_df_2[affi_df_2['pdb_id']==mol_2_name]['-logAffi'].values[0]
                else:
                    mol_2_affi = affi_df_2[affi_df_2['unique_identify']==mol_2_name]['-logAffi'].values[0]
                mol_2_smiles = Chem.MolToSmiles(mols_2[j])
                lst.append([mol_name, mol_affi, mol_smiles, mol_2_name, mol_2_affi, mol_2_smiles, simi])
        max_simi = max(simi_list)
        max_simi_for_hold_2019_mols.append(max_simi)
    # out_dir = f'/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019/4-evaluation/test_on_PDBbind_hold_out_2019/similarity/{set_name_2}_in_{set_name_1}'
    # if not Path(out_dir).exists():
    #     Path(out_dir).mkdir(parents=True)
    # simi_8_df = pd.DataFrame(lst, columns=cols)
    # simi_8_df.to_csv(f'{out_dir}/simi_0.8.csv', sep='\t', index=False)
    print(f'For {set_name_1}, there are {max_simi_for_hold_2019_mols.count(1)} / {len(max_simi_for_hold_2019_mols)} PDB IDs same as(similarity=1, not considering chirality) {set_name_2}.')
    print(f'similarity > 0.8: {len([n for n in max_simi_for_hold_2019_mols if n>0.8])} / {len(max_simi_for_hold_2019_mols)}')
    return max_simi_for_hold_2019_mols

index_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PLANet_smiles_df = pd.read_csv(f'{index_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv', sep='\t')


PDBbind_v18_subset_df = pd.read_csv('index_rm_all_simi_1/PDBbind_v18_subset_rm_simi_1.csv', sep='\t')
PDBbind_v18_subset_mols, PDBbind_v18_subset_mol_fps = obtain_training_fps(PDBbind_v18_subset_df, 'PDBbind_v18_subset')

PLANet_v18_df = pd.read_csv('index_rm_all_simi_1/PLANet_v18_rm_simi_1.csv', sep='\t')
PLANet_v18_mols, PLANet_v18_mol_fps = obtain_training_fps(PLANet_v18_df, 'PLANet_v18')

PIPUP_v18_df = pd.read_csv('index_rm_all_simi_1/PDBbind_v18_subset_union_PLANet_rm_simi_1.csv', sep='\t')
PIPUP_v18_mols, PIPUP_v18_mol_fps = obtain_training_fps(PIPUP_v18_df, 'PIPUP_v18')

PDBbind_hold_out_mols, PDBbind_hold_out_mol_fps = obtain_training_fps(PDBbind_hold, 'PDBbind_hold_out')
PDBbind_hold_out_mol_names = [mol.GetProp('_Name') for mol in PDBbind_hold_out_mols]

max_simi_for_PDBbind_hold_2019_test_PDBbind_v18_subset = calculate_simi(PDBbind_hold_out_mols, PDBbind_hold_out_mol_fps, PDBbind_hold, PDBbind_v18_subset_mols, PDBbind_v18_subset_mol_fps, PDBbind_v18_subset_df, f'PDBbind_hold_out_2019', 'PDBbind_v18_subset')
max_simi_for_PDBbind_hold_2019_test_Uw = calculate_simi(PDBbind_hold_out_mols, PDBbind_hold_out_mol_fps, PDBbind_hold, PLANet_v18_mols, PLANet_v18_mol_fps, PLANet_v18_df, f'PDBbind_hold_out_2019', 'PLANet_v18')
max_simi_for_PDBbind_hold_2019_test_PIPUP = calculate_simi(PDBbind_hold_out_mols, PDBbind_hold_out_mol_fps, PDBbind_hold, PIPUP_v18_mols, PIPUP_v18_mol_fps, PIPUP_v18_df, f'PDBbind_hold_out_2019', 'PIPUP_v18')

if not Path('similarity').exists():
    Path('similarity').mkdir()
max_simi_distribution = pd.DataFrame({"pdb_id": PDBbind_hold_out_mol_names, "PDBbind_v18_subset": max_simi_for_PDBbind_hold_2019_test_PDBbind_v18_subset, "PLANet_v18": max_simi_for_PDBbind_hold_2019_test_Uw, "PDBbind_v18_subset_union_PLANet_v18": max_simi_for_PDBbind_hold_2019_test_PIPUP})
max_simi_distribution.to_csv(f'similarity/all_6_models_max_simi_distribution.csv', sep='\t', index=False)
