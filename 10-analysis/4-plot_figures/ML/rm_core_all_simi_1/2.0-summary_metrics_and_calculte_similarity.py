from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

if not Path('output').exists():
    Path('output').mkdir()

# 1. sum.csv
wrkdir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim'
label_to_res_dict = defaultdict(list)
log_dir_names = ['PDBbind_minimized_intersected_Uw','PLANet_Uw', 'PDBbind_minimized_intersected_Uw_union_Uw'] # plot order

mdl_types = ['complex_6A', 'lig_alone']
test_types = ['valid', 'train', 'test', 'CASF_v16_minimized', 'CASF_v16_intersected_Uw_minimized', 'CASF_v16_original', 'CASF_v16_intersected_Uw_original'] #change by the order in log

# label_to_res_dict = defaultdict(list)
for log_dir_name in log_dir_names:
    for mdl_type in mdl_types:
        if log_dir_name in ['PDBbind_minimized', 'PDBbind_minimized_intersected_Uw']:
            log_dir =  f'{wrkdir}/PDBbind/pdbbind_v2019/minimized/2-train/true_lig_alone/test/diff_split/rm_core_ids/{log_dir_name}/{mdl_type}/log'
        else:
            log_dir = f'{wrkdir}/3-test/scripts/true_lig_alone_modify_dists/diff_split/rm_core_ids/{log_dir_name}/{mdl_type}/log/'

        log_files = [str(p) for p in list(Path(log_dir).glob('*log'))]
        log_files.sort()
        for i, log_f in enumerate(log_files):
            with open(log_f, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if f'Performance on' in line:
                    if line.split()[2] in test_types:
                        R2 = float(line.split(',')[0].split(':')[2])
                        mae = float(line.split(',')[1].split(':')[1])
                        mse = float(line.split(',')[2].split(':')[1])
                        rmse = np.sqrt(mse)
                        pearsonr = float(line.split(',')[3].split('(')[1])
                        spearmanr =float(line.split(',')[5].split('=')[1])
                        label_to_res_dict[f'{log_dir_name}_{mdl_type}_{i+1}_{line.split()[2]}']=[R2, mae, mse, rmse, pearsonr, spearmanr]

sum_df = pd.DataFrame.from_dict(label_to_res_dict, orient='index', columns=['R2', 'mae', 'mse', 'rmse', 'pearsonr', 'spearmanr']).reset_index()
sum_df.rename(columns={"index": "model_names_test_type"}, inplace=True)
sum_df['model_names'] = ['_'.join(m.split('_')[:-3]) if 'CASF_v16_minimized' in m or 'CASF_v16_original' in m else ('_'.join(m.split('_')[:-5]) if 'Uw_minimized' in m or 'Uw_original' in m else '_'.join(m.split('_')[:-1])) for m in sum_df['model_names_test_type']]
sum_df['test_type'] = ['_'.join(m.split('_')[-3:]) if 'CASF_v16_minimized' in m or 'CASF_v16_original' in m else ('_'.join(m.split('_')[-5:]) if 'Uw_minimized' in m or 'Uw_original' in m else m.split('_')[-1]) for m in sum_df['model_names_test_type']]
# sum_df['model_names'] = ['_'.join(m.split('_')[:-3]) if 'CASF_v16_minimized' in m else ('_'.join(m.split('_')[:-5]) if 'Uw_minimized' in m else '_'.join(m.split('_')[:-1])) for m in sum_df['model_names_test_type']]
# sum_df['test_type'] = ['_'.join(m.split('_')[-3:]) if 'CASF_v16_minimized' in m else ('_'.join(m.split('_')[-5:]) if 'Uw_minimized' in m else m.split('_')[-1]) for m in sum_df['model_names_test_type']]
sum_df['dataset'] = sum_df['model_names'].str.rsplit('_', n=3).str[0]
sum_df['model_type'] = ['complex' if 'complex' in m else 'ligand_alone' for m in sum_df['model_names']]
sum_df['dataset'] = ['PDBbind' if n=='PDBbind_minimized' else ('PIP') if n=='PDBbind_minimized_intersected_Uw' else 'PIPUP' if n=='PDBbind_minimized_intersected_Uw_union_Uw' else 'PLANet' if n=='PLANet_Uw' else 'PUP' if n=='PDBbind_minimized_union_Uw' else n for n in sum_df['dataset']]
sum_df.to_csv('output/sum.csv', sep='\t', index=False)


# 2. CIP_cmx
PLANet_cmx_1 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/rm_core_ids/PLANet_Uw/complex_6A/1/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PLANet_cmx_1.rename(columns={'y_pred':'PLANet_cmx_1'}, inplace=True)
PLANet_cmx_2 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/rm_core_ids/PLANet_Uw/complex_6A/2/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PLANet_cmx_2.rename(columns={'y_pred':'PLANet_cmx_2'}, inplace=True)
PLANet_cmx_3 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/rm_core_ids/PLANet_Uw/complex_6A/3/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PLANet_cmx_3.rename(columns={'y_pred':'PLANet_cmx_3'}, inplace=True)
PLANet_cmx_4 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/rm_core_ids/PLANet_Uw/complex_6A/4/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PLANet_cmx_4.rename(columns={'y_pred':'PLANet_cmx_4'}, inplace=True)
PLANet_cmx_5 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/rm_core_ids/PLANet_Uw/complex_6A/5/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PLANet_cmx_5.rename(columns={'y_pred':'PLANet_cmx_5'}, inplace=True)

PIP_cmx_1 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/test_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/1/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PIP_cmx_1.rename(columns={'y_pred':'PIP_cmx_1'}, inplace=True)
PIP_cmx_2 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/test_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/2/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PIP_cmx_2.rename(columns={'y_pred':'PIP_cmx_2'}, inplace=True)
PIP_cmx_3 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/test_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/3/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PIP_cmx_3.rename(columns={'y_pred':'PIP_cmx_3'}, inplace=True)
PIP_cmx_4 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/test_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/4/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PIP_cmx_4.rename(columns={'y_pred':'PIP_cmx_4'}, inplace=True)
PIP_cmx_5 = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/test_result/diff_split/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/5/CASF_v16_intersected_Uw_minimized.csv', sep='\t')
PIP_cmx_5.rename(columns={'y_pred':'PIP_cmx_5'}, inplace=True)

CIP_cmx = PLANet_cmx_1.merge(PLANet_cmx_2, on=['unique_identify', 'y_true']).merge(PLANet_cmx_3, on=['unique_identify', 'y_true']).merge(PLANet_cmx_4, on=['unique_identify', 'y_true']).merge(PLANet_cmx_5, on=['unique_identify', 'y_true']).merge(PIP_cmx_1, on=['unique_identify', 'y_true']).merge(PIP_cmx_2, on=['unique_identify', 'y_true']).merge(PIP_cmx_3, on=['unique_identify', 'y_true']).merge(PIP_cmx_4, on=['unique_identify', 'y_true']).merge(PIP_cmx_5, on=['unique_identify', 'y_true'])
CIP_cmx['PLANet_cmx_mean'] = CIP_cmx[['PLANet_cmx_1', 'PLANet_cmx_2', 'PLANet_cmx_3', 'PLANet_cmx_4', 'PLANet_cmx_5']].mean(axis=1)
CIP_cmx['PIP_cmx_mean'] = CIP_cmx[['PIP_cmx_1', 'PIP_cmx_2', 'PIP_cmx_3', 'PIP_cmx_4', 'PIP_cmx_5']].mean(axis=1)
CIP_cmx.to_csv('output/Core_inter_Uw_scatter_PLANet_vs_PIP_cmx_mean.csv', sep='\t', index=False)


# 3. similarity
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem
from rdkit import DataStructs

def obtain_fps(data_df, dataset_name):
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
    # out_dir = f'{wrkdir}/4-similarity_analysis/rm_core_ids/core_intersected_Uw/{set_name_2}_{set_name_1}' # run at 20220825
    # out_dir = f'/pubhome/xli02/project/PLIM/analysis/20220829_paper/ML/rm_core_all_simi_1/similarity/{set_name_2}_{set_name_1}'
    # if not Path(out_dir).exists():
    #     Path(out_dir).mkdir()
    # simi_1_df = pd.DataFrame(lst, columns=cols)
    # simi_1_df.to_csv(f'{out_dir}/simi_1.csv', sep='\t', index=False)
    return max_simi_for_mols_1

# wrkdir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim'
pdbbind_dir = '/pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019'

index_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PLANet_smiles_df = pd.read_csv(f'{index_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv', sep='\t')

# core_intersected_Uw
core_intersected_Uw_df = pd.read_csv('index/core_intersected_Uw.csv', sep='\t')
core_intersected_Uw_mols, core_intersected_Uw_mol_fps = obtain_fps(core_intersected_Uw_df, 'CASF_v16_intersected_Uw')
core_intersected_Uw_mol_names = [mol.GetProp('_Name') for mol in core_intersected_Uw_mols]

# Uw
Uw_df = pd.read_csv('index/PLANet_Uw_remove_core_set_ids.csv', sep='\t')
Uw_mols, Uw_mol_fps = obtain_fps(Uw_df, f'PLANet_Uw') # 4min

# PDBbind_minimized_subset
PDBbind_intersected_Uw_df = pd.read_csv('index/PDBbind_minimized_intersected_Uw_rm_core_ids.csv', sep='\t')
PDBbind_intersected_Uw_mols, PDBbind_intersected_Uw_mol_fps = obtain_fps(PDBbind_intersected_Uw_df, 'PDBbind_minimized_intersected_Uw')

# PDBbind_minimized_subset_union_Uw
PDBbind_intersected_Uw_union_Uw_df = pd.read_csv('index/PDBbind_minimized_intersected_Uw_union_Uw_rm_core_ids.csv', sep='\t')
PDBbind_intersected_Uw_union_Uw_mols, PDBbind_intersected_Uw_union_Uw_mol_fps = obtain_fps(PDBbind_intersected_Uw_union_Uw_df, f'PDBbind_minimized_intersected_Uw_union_Uw') # 5min

max_simi_for_core_intersected_Uw_test_Uw = calculate_simi(core_intersected_Uw_mols, core_intersected_Uw_mol_fps, core_intersected_Uw_df, Uw_mols, Uw_mol_fps, Uw_df, 'CASF_v16_intersected_Uw_test', 'Uw')

max_simi_for_core_intersected_Uw_test_intersected_Uw = calculate_simi(core_intersected_Uw_mols, core_intersected_Uw_mol_fps, core_intersected_Uw_df, PDBbind_intersected_Uw_mols, PDBbind_intersected_Uw_mol_fps, PDBbind_intersected_Uw_df, 'CASF_v16_intersected_Uw_test', 'PDBbind_minimized_intersected_Uw')

max_simi_for_core_intersected_Uw_test_PDBbind_intersected_Uw_union_Uw = calculate_simi(core_intersected_Uw_mols, core_intersected_Uw_mol_fps, core_intersected_Uw_df, PDBbind_intersected_Uw_union_Uw_mols, PDBbind_intersected_Uw_union_Uw_mol_fps, PDBbind_intersected_Uw_union_Uw_df, 'CASF_v16_intersected_Uw_test', 'PDBbind_minimized_intersected_Uw_union_Uw')

if not Path('similarity').exists():
    Path('similarity').mkdir()
max_simi_distribution = pd.DataFrame({'pdb_id':core_intersected_Uw_mol_names, "PDBbind_minimized_intersected_Uw": max_simi_for_core_intersected_Uw_test_intersected_Uw, "PLANet_Uw": max_simi_for_core_intersected_Uw_test_Uw, "PDBbind_minimized_intersected_Uw_union_Uw": max_simi_for_core_intersected_Uw_test_PDBbind_intersected_Uw_union_Uw})
max_simi_distribution.to_csv(f'similarity/all_models_max_simi_distribution.csv', sep='\t', index=False)

