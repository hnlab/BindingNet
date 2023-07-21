'''
find mmp-cliffs using mmpdb (PLANet_all_ML as index for more crystal templates, and PLANet_Uw_SAR(activity.csv) as activity)
1. strandardize input SMILES(NOT including crystal ligands)
2. identify mmp uding mmpdb (it seems fragmentaion is not exhaustive)
    fragment -> index
    one cut only
3. obtain activities
4. mmp cliffs: 10 fold
5. mmp cliffs with the same crystal template
'''

import os
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol

wrkdir = '/pubhome/xli02/project/PLIM/activity_cliff'
PLANet_all_dealt = pd.read_csv('/pubhome/xli02/project/PLIM/v2019_dataset/index/20220524_other_files/PLANet_all_dealt.csv', sep='\t')
activity_df = pd.read_csv(f'{wrkdir}/activity.csv', sep='\t')

failed_targets = []
no_mmp_cliffs_targets = []
no_mmp_cliffs_with_crystal_tps_targets = []
# target_id = 'CHEMBL3072'
target_to_cat = {}
for target_id in set(PLANet_all_dealt['Target_chembl_id']):
    target_dir = f'{wrkdir}/results/{target_id}'
    if not Path.exists(Path(target_dir)):
        Path.mkdir(Path(target_dir))

    target_df = PLANet_all_dealt[PLANet_all_dealt['Target_chembl_id']==target_id].copy()
    # 1. standardize input SMILES(NOT including crystal ligands)
    smi_df = target_df[['Similar_compnd_smiles', 'Similar_compnd_name']].drop_duplicates()
    smi_df.reset_index(drop=True, inplace=True)
    if len(smi_df) < 2:
        print(f'For target {target_id}, there are less than two compounds, thus no MMP.')
        failed_targets.append(target_id)
        continue

    remover = SaltRemover()
    param = rdMolStandardize.CleanupParameters() # rdkit version should be newer than 2020.09
    param.tautomerRemoveSp3Stereo = False
    param.tautomerRemoveBondStereo = False
    enumerator = rdMolStandardize.TautomerEnumerator(param)

    for row in smi_df.itertuples():
        m = Chem.MolFromSmiles(row.Similar_compnd_smiles)
        m_nosalt = remover.StripMol(m, dontRemoveEverything=True) # remove salt, like CC(=O)O.[Na] -> CC(=O)O
        neutralize_atoms(m_nosalt)
        m_cano = enumerator.Canonicalize(m_nosalt)
        m_cano_smi = Chem.MolToSmiles(m_cano)
        smi_df.loc[row.Index, 'standardized_smi'] = m_cano_smi

    smi_f = f'{target_dir}/{target_id}_standardized.smi'
    smi_df[['standardized_smi', 'Similar_compnd_name']].to_csv(smi_f, sep='\t', index=False, header=False)

    # 2. identify mmp uding mmpdb
    frag_f = f'{target_dir}/{target_id}_para.frag'
    mmpdb_out = f'{target_dir}/{target_id}_para_out.csv'
    os.system(f'mmpdb fragment {smi_f} --num-cuts 1 -o {frag_f}')
    os.system(f"mmpdb index {frag_f} -s --max-variable-ratio 0.33 --max-heavies-transf 8 -o {mmpdb_out} --out 'csv'")
    mmpdb_out_df = pd.read_csv(mmpdb_out, sep='\t', names=['smi_1', 'smi_2', 'mol1', 'mol2', 'transformation', 'core'])
    if len(mmpdb_out_df) == 0:
        print(f'For target {target_id}, there is no MMP generated after mmpdb.')
        failed_targets.append(target_id)
        continue
    mmpdb_out_df['NHA_core'] = [Chem.MolFromSmiles(core).GetNumAtoms()-1 for core in mmpdb_out_df['core']]
    mmpdb_max_core_only_df = mmpdb_out_df.sort_values(by='NHA_core', ascending=False).groupby(['mol1', 'mol2']).head(1).copy()

    # 3. obtain activities: compounds in one MMP must have Ki/Kd in same/different assay, or IC50/EC50 in same assay
    mmps_list = np.sort(mmpdb_max_core_only_df[['mol1', 'mol2']], axis=1).tolist()
    mmps = [list(m) for m in set(tuple(mmp) for mmp in mmps_list)]

    target_compd_activity_df = activity_df[(activity_df['target_chembl_id'] == target_id) & (activity_df['compound'].isin(list(set(mmpdb_max_core_only_df['mol1']))))]
    mmp_assay_list = []
    for as_tp in ['Kd', 'Ki']:
        acti = target_compd_activity_df[target_compd_activity_df['activity_type'] == as_tp].copy()
        for m in mmps:
            if set(m).issubset(set(list(acti['compound']))):
                assay_ids = [acti[acti['compound']==cpd]['assay_chembl_id'].values[0] for cpd in m]
                affinities = [acti[acti['compound']==cpd]['pAffi'].values[0] for cpd in m]
                mmp_assay_list.append(m + [target_id] + assay_ids + [as_tp] + affinities)
    other_activity_type = target_compd_activity_df[~target_compd_activity_df['activity_type'].isin(['Ki', 'Kd'])].copy()
    grouped = other_activity_type.groupby('assay_chembl_id')
    for assay, grp in grouped:
        for m in mmps:
            if set(m).issubset(set(list(grp['compound']))):
                affinities = [grp[grp['compound']==cpd]['pAffi'].values[0] for cpd in m]
                activity_type = grp[grp['compound']==m[0]]['activity_type'].values[0] # whether same assay id have same activity_type?: YES! CHEMBL1118768_CHEMBL1084623
                mmp_assay_list.append(m+ [target_id, assay, assay, activity_type] + affinities)
    mmp_assay_df = pd.DataFrame(mmp_assay_list, columns =['mol1', 'mol2', 'target_chembl_id', 'assay_chembl_id1', 'assay_chembl_id2', 'activity_type', 'pAffi_1', 'pAffi_2'])
    if len(mmp_assay_df) == 0:
        print(f'For target {target_id}, there is no MMP-forming compounds in the same assay or same Ki/Kd.')
        failed_targets.append(target_id)
        continue

    # unique activity type for one mmp: Kd > Ki > IC50 > EC50
    only_one_type_list = []
    for name, mmp_grp in mmp_assay_df.groupby(['mol1', 'mol2']):
        if 'Kd' in set(mmp_grp['activity_type']):
            acti_df = mmp_grp.query('activity_type == "Kd"')
        elif 'Ki' in set(mmp_grp['activity_type']):
            acti_df = mmp_grp.query('activity_type == "Ki"')
        elif 'IC50' in set(mmp_grp['activity_type']):
            acti_df = mmp_grp.query('activity_type == "IC50"')
        else:
            acti_df = mmp_grp.query('activity_type == "EC50"')
        only_one_type_list.append(acti_df)
    only_one_type_df=pd.concat(only_one_type_list, ignore_index=True)
    mmps_info_with_unique_assay_type = f'{target_dir}/{target_id}_one_way_with_unique_assay_type_may_exsit_multiple_assays.csv'
    only_one_type_df.to_csv(mmps_info_with_unique_assay_type, sep='\t', index=False)

    # multiple assays within unique activity type for one mmp
    # potency inconsistent: remove
    unique_activity_list = []
    for name, grouped_df in only_one_type_df.groupby(['mol1', 'mol2']):
        acti_df = grouped_df.copy().reset_index(drop=True)
        if len(acti_df) > 1:
            acti_df.loc[:, 'unique_assay'] = False
            if (acti_df['pAffi_1'].max() - acti_df['pAffi_1'].min() > 1) or (acti_df['pAffi_2'].max() - acti_df['pAffi_2'].min() > 1):  # compounds with multiple assays and potency not consistent for any compound in mmp
                print(f'MMP {name} has multiple assays with the same assay types, and potency of any compound is inconsistent, remove this MMP.')
                continue
            else: # median pAffi
                print(f'MMP {name} has multiple assays with the same assay types, but potency of both compounds is consistent, use median value as pAffi.')
                acti_df.loc[:, 'pAffi_1'] = acti_df['pAffi_1'].median()
                acti_df.loc[:, 'pAffi_2'] = acti_df['pAffi_2'].median()
                acti_df.loc[:, 'delta_pAffi'] = np.abs(acti_df['pAffi_1'] - acti_df['pAffi_2'])
        else:
            acti_df.loc[:, 'unique_assay'] = True
            acti_df.loc[:, 'delta_pAffi'] = np.abs(acti_df['pAffi_1'] - acti_df['pAffi_2']).values[0]
        unique_activity = acti_df[['mol1', 'mol2', 'target_chembl_id', 'activity_type', 'unique_assay', 'pAffi_1', 'pAffi_2', 'delta_pAffi']].drop_duplicates()
        unique_activity_list.append(unique_activity)
    if len(unique_activity_list) == 0:
        print(f'For target {target_id}, there is no MMPs with both compounds having consistent potency.')
        failed_targets.append(target_id)
        continue
    unique_activity_df=pd.concat(unique_activity_list, ignore_index=True)
    mmps_info_with_unique_activity = pd.merge(unique_activity_df, mmpdb_max_core_only_df, on=['mol1', 'mol2'])
    mmps_info_with_unique_activity_pAffi_medianed = f'{target_dir}/{target_id}_one_way_with_unique_activity_pAffi_medianed.csv'
    mmps_info_with_unique_activity.to_csv(mmps_info_with_unique_activity_pAffi_medianed, sep='\t', index=False)

    # generate a table of final pAffi_medianed with all MMP-forming compounds For Cytoscape Node Color (!! one compound participating in multiple MMPs could have multiple pAffi)
    mmp_compounds_pAffi_medianed = pd.concat([mmps_info_with_unique_activity[['mol1', 'pAffi_1']].rename(columns={'mol1': 'mmp_mol', 'pAffi_1':'mmp_pAffi'}), mmps_info_with_unique_activity[['mol2', 'pAffi_2']].rename(columns={'mol2': 'mmp_mol', 'pAffi_2':'mmp_pAffi'})]).drop_duplicates()
    mmp_compounds_pAffi_medianed_f = f'{target_dir}/{target_id}_mmp_cpds_pAffi_medianed.csv'
    mmp_compounds_pAffi_medianed.to_csv(mmp_compounds_pAffi_medianed_f, index=False)
    multiple_activity_values_compounds = set(mmp_compounds_pAffi_medianed.groupby('mmp_mol').filter(lambda x: len(x)>1)['mmp_mol'])
    if len(multiple_activity_values_compounds) != 0:
        print(f'{multiple_activity_values_compounds} have different pAffi in different MMPs.')

    # 4. mmp cliffs: 10 fold
    activity_cliff_mmp_f = f'{target_dir}/{target_id}_one_way_mmp_cliffs.csv'
    activity_cliff_mmp = mmps_info_with_unique_activity[mmps_info_with_unique_activity['delta_pAffi'] > 1][['mol1', 'mol2', 'target_chembl_id', 'activity_type', 'unique_assay', 'pAffi_1', 'pAffi_2', 'delta_pAffi', 'smi_1', 'smi_2', 'transformation', 'core', 'NHA_core']].drop_duplicates()
    if len(activity_cliff_mmp) == 0:
        print(f'For target {target_id}, there is no MMP-cliffs, with 10-fold difference potency.')
        no_mmp_cliffs_targets.append(target_id)
        continue
    activity_cliff_mmp.to_csv(activity_cliff_mmp_f, index=False,float_format='%.2f')

    mmp_cliffs_list = np.sort(activity_cliff_mmp[['mol1', 'mol2']], axis=1).tolist()
    mmp_cliffs = [list(m) for m in set(tuple(mmp) for mmp in mmp_cliffs_list)]
    # len(mmp_cliffs)
    mmpdb_max_core_only_df.reset_index(drop=True, inplace=True)
    mmpdb_max_core_only_df.loc[:, 'mmp_cliff'] = False
    for row in mmpdb_max_core_only_df.itertuples():
        for mmp_cliff in mmp_cliffs:
            if row.mol1 in mmp_cliff and row.mol2 in mmp_cliff:
                mmpdb_max_core_only_df.loc[row.Index, 'mmp_cliff'] = True

    # 5. mmp cliffs with the same crystal template
    mmp_cry_tplt_list = []
    cry_tplts = list(set(target_df['Cry_lig_name']))
    for cry_tplt in cry_tplts:
        cry_tplt_df = target_df[target_df['Cry_lig_name'] == cry_tplt].copy()
        for mmp_cliff in mmp_cliffs:
            if set(mmp_cliff).issubset(set(list(cry_tplt_df['Similar_compnd_name']))):
                total_deltas = [cry_tplt_df[cry_tplt_df['Similar_compnd_name']==cpd]['Total_delta'].values[0] for cpd in mmp_cliff]
                lig_deltas = [cry_tplt_df[cry_tplt_df['Similar_compnd_name']==cpd]['Lig_delta'].values[0] for cpd in mmp_cliff]
                core_RMSDs = [cry_tplt_df[cry_tplt_df['Similar_compnd_name']==cpd]['Core_RMSD'].values[0] for cpd in mmp_cliff]
                MCS_smart = [cry_tplt_df[cry_tplt_df['Similar_compnd_name']==cpd]['MCS_smarts'].values[0] for cpd in mmp_cliff]
                worst_total_delta = max(total_deltas)
                worst_lig_delta = min(lig_deltas)
                worst_core_RMSD = max(core_RMSDs)
                mmp_cry_tplt_list.append(mmp_cliff + [target_id, cry_tplt, worst_total_delta, worst_lig_delta, worst_core_RMSD] + total_deltas + lig_deltas + core_RMSDs + MCS_smart)
    mmp_all_cry_tplts_df = pd.DataFrame(mmp_cry_tplt_list, columns =['mol1', 'mol2', 'target_chembl_id', 'cry_lig_name', 'worst_Total_delta', 'worst_Lig_delta', 'worst_Core_RMSD', 'Total_delta_1', 'Total_delta_2', 'Lig_delta_1', 'Lig_delta_2', 'Core_RMSD_1', 'Core_RMSD_2', 'MCS_smarts_1', 'MCS_smarts_2'])
    if len(mmp_all_cry_tplts_df) == 0:
        print(f'For target {target_id}, there is no MMP-cliffs with the same crystal templates.')
        no_mmp_cliffs_with_crystal_tps_targets.append(target_id)
        continue
    mmp_best_cry_tplt_df = mmp_all_cry_tplts_df.groupby(['mol1', 'mol2']).apply(lambda x: x.sort_values(by=['worst_Total_delta', 'worst_Lig_delta', 'worst_Core_RMSD'], ascending=[True, False, True]).head(1)).copy()
    mmp_all_cry_tplts_df.to_csv(f'{target_dir}/{target_id}_one_way_mmp_cliffs_all_possible_cry_templates.csv', sep='\t', index=False)

    mmp_cliff_to_cry_tplt = {}
    for row in mmp_best_cry_tplt_df.itertuples():
        mmp_cliff_to_cry_tplt[tuple([row.mol1, row.mol2])] = row.cry_lig_name
        mmp_cliff_to_cry_tplt[tuple([row.mol2, row.mol1])] = row.cry_lig_name

    mmpdb_max_core_only_df.reset_index(drop=True, inplace=True)
    mmpdb_max_core_only_df.loc[:, 'crystal_ligand'] = ''
    for row in mmpdb_max_core_only_df.itertuples():
        if tuple([row.mol1, row.mol2]) in mmp_cliff_to_cry_tplt.keys():
            mmpdb_max_core_only_df.loc[row.Index, 'crystal_ligand'] = mmp_cliff_to_cry_tplt[tuple([row.mol1, row.mol2])]
    mmpdb_max_core_only = f'{target_dir}/{target_id}_para_out_unique_trans_per_pair_with_cliffs_and_crytps.csv'
    mmpdb_max_core_only_df['target'] = target_id
    mmpdb_max_core_only_df.to_csv(mmpdb_max_core_only, index=False)

    # 6. plot distribution of potency / delta_pAffi
    from matplotlib import pyplot as plt
    import seaborn as sns

    fig, ax= plt.subplots(figsize=(4,6))
    ax.set_title(f'pAffi of MMP-forming compounds in target {target_id} (N={len(mmp_compounds_pAffi_medianed)})')
    sns.boxplot(y=mmp_compounds_pAffi_medianed["mmp_pAffi"])
    plt.savefig(f'{target_dir}/{target_id}_mmp_cpds_pAffi_medianed.png', dpi=300, bbox_inches='tight')
    plt.close()

    q1, q3 = np.percentile(mmp_compounds_pAffi_medianed["mmp_pAffi"], [25,75])
    iqr = q3 - q1
    if iqr<1:
        target_to_cat[target_id] = 'CAT1'
    elif iqr<2:
        target_to_cat[target_id] = 'CAT2'
    else:
        target_to_cat[target_id] = 'CAT3'

    # target_set-dependent cutoff
    # np.mean(mmps_info_with_unique_activity["delta_pAffi"]) + 2 * np.std(mmps_info_with_unique_activity["delta_pAffi"])

    fig, ax= plt.subplots(figsize=(4,6))
    ax.set_title(f'delta_pAffi of MMPs in target {target_id} (N={len(mmps_info_with_unique_activity)})')
    sns.boxplot(y=mmps_info_with_unique_activity["delta_pAffi"])
    plt.savefig(f'{target_dir}/{target_id}_mmps_delta_pAffi.png', dpi=300, bbox_inches='tight')
    plt.close()

failed_1_df = pd.DataFrame(failed_targets, columns=['target_chembl_id'])
failed_1_df['failing_reason'] = 'before_cliffs'
failed_2_df = pd.DataFrame(no_mmp_cliffs_targets, columns=['target_chembl_id'])
failed_2_df['failing_reason'] = 'no_cliffs'
failed_3_df = pd.DataFrame(no_mmp_cliffs_with_crystal_tps_targets, columns=['target_chembl_id'])
failed_3_df['failing_reason'] = 'no_cry_tps'
failed_df = pd.concat([failed_1_df, failed_2_df, failed_3_df])
failed_df.to_csv(f'{wrkdir}/results/index/failed_target.csv', sep='\t', index=False)

target_cat = pd.DataFrame.from_dict(target_to_cat, orient='index', columns=['categories']).reset_index().rename(columns={'index':'target_chemb_id'})
target_cat.to_csv(f'{wrkdir}/results/index/target_category.csv', sep='\t', index=False)

print(f'There are {len(failed_targets)} targets failed before mmp-cliffs, e.g., while finding mmp, or finding consistent affinities for mmp-forming compounds.')
print(f'There are {len(no_mmp_cliffs_targets)} targets failed while finding cliffs.')
print(f'There are {len(no_mmp_cliffs_with_crystal_tps_targets)} targets failed while finding crystal structures.')
