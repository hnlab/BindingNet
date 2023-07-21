import pandas as pd

wrkdir = '/home/lixl/Documents/Project/ChEMBL-scaffold/12-web_server/1-mk_table'
dataset_dir = '/data/lixl/from_x254/ChEMBL-scaffold/'

# 1.PLANet_Uw
Uw_for_SAR_info = f'{wrkdir}/PLANet_Uw_for_SAR_rm_NA_unique.csv'
Uw_for_SAR_info_df = pd.read_csv(Uw_for_SAR_info, sep='\t')
# len(Uw_for_SAR_info_df) #105920

## 1.1 crylig.csv
cry_lig_info = Uw_for_SAR_info_df[['Cry_lig_name', 'Cry_lig_smiles', 'Cry_lig_an']]
cry_lig_info_drop_dup = cry_lig_info.drop_duplicates().copy()
cry_lig_info_drop_dup.rename(columns={"Cry_lig_name": "pdbid", "Cry_lig_smiles": "crylig_smiles", "Cry_lig_an": "crylig_heavy_atom_number"}, inplace=True)
cry_lig_info_drop_dup.to_csv(f'{wrkdir}/planet_Uw/crylig.csv', sep='\t', index=False)
# len(cry_lig_info_drop_dup) #5908
# cry_lig_info_drop_dup.crylig_smiles.map(lambda x: len(x)).max() #196

## 1.2 compound.csv
compound_info = Uw_for_SAR_info_df[['Similar_compnd_name', 'Similar_compnd_smiles', 'Similar_compnd_an', 'MolWt']]
compound_info_rounded = compound_info.round({"MolWt":1})
compound_info_rounded_drop_dup = compound_info_rounded.drop_duplicates().copy()
compound_info_rounded_drop_dup.rename(columns={'Similar_compnd_name': 'compound', 'Similar_compnd_smiles': 'compound_smiles', 'Similar_compnd_an': 'compound_heavy_atom_number', 'MolWt': 'compound_Mw'}, inplace=True)
compound_info_rounded_drop_dup.to_csv(f'{wrkdir}/planet_Uw/compound.csv', sep='\t', index=False)
# len(compound_info_drop_dup) #65924

## 1.3 activity
activity_df = Uw_for_SAR_info_df[['Target_chembl_id', 'Similar_compnd_name', 'standard_type', 'standard_relation', 'standard_value', 'standard_units', 'assay_chembl_id', '-logAffi']]
activity_df_logAffi_rounded = activity_df.round({"-logAffi":2}).copy()
activity_df_logAffi_rounded_drop_dup = activity_df_logAffi_rounded.drop_duplicates().copy()
activity_df_logAffi_rounded_drop_dup.rename(columns={'Target_chembl_id': 'target_chembl_id', 'Similar_compnd_name':'compound', 'standard_type':'activity_type', 'standard_relation':'activity_relation', 'standard_value': 'activity_value', 'standard_units':'activity_units', '-logAffi': 'pAffi'}, inplace=True)
activity_df_logAffi_rounded_drop_dup.to_csv(f'{wrkdir}/planet_Uw/activity.csv', sep='\t', index=False)
# len(activity_df_logAffi_rounded_drop_dup) # 105920

## 1.4 assay
assay_df = Uw_for_SAR_info_df[['assay_chembl_id', 'assay_type']]
assay_df_drop_dup = assay_df.drop_duplicates().copy()
assay_df_drop_dup.to_csv(f'{wrkdir}/planet_Uw/assay.csv', sep='\t', index=False)
# len(assay_df_drop_dup) #16434

## 1.5 complex
# Uw_complex_df = Uw_for_SAR_info_df[['Target_chembl_id', 'Cry_lig_name', 'Similar_compnd_name', 'Part_fix', 'Total_sampled_num', 'Total_delta', 'Lig_delta', 'Core_RMSD']] # 'part_fix'放入crylig_compound中
Uw_complex_df = Uw_for_SAR_info_df[['Target_chembl_id', 'Cry_lig_name', 'Similar_compnd_name', 'Total_sampled_num', 'Total_delta', 'Lig_delta', 'Core_RMSD']]
Uw_complex_df_coreRMSD_rounded = Uw_complex_df.round({'Core_RMSD':2})
Uw_complex_df_coreRMSD_rounded_drop_dup = Uw_complex_df_coreRMSD_rounded.drop_duplicates().copy()
# Uw_complex_df_coreRMSD_rounded_drop_dup.rename(columns={'Target_chembl_id':'target_chembl_id', 'Cry_lig_name':'pdbid', 'Similar_compnd_name': 'compound', 'Part_fix':'fix_part_core','Total_sampled_num': 'total_sampled_num', 'Total_delta':'calculated_binding_energy', 'Lig_delta':'calculated_delta_lig_conform_energy', 'Core_RMSD': 'core_RMSD'}, inplace=True)
Uw_complex_df_coreRMSD_rounded_drop_dup.rename(columns={'Target_chembl_id':'target_chembl_id', 'Cry_lig_name':'pdbid', 'Similar_compnd_name': 'compound', 'Total_sampled_num': 'total_sampled_num', 'Total_delta':'calculated_binding_energy', 'Lig_delta':'calculated_delta_lig_conform_energy', 'Core_RMSD': 'core_RMSD'}, inplace=True)
Uw_complex_df_coreRMSD_rounded_drop_dup.to_csv(f'{wrkdir}/planet_Uw/complex.csv', sep='\t', index=False)
# len(Uw_complex_df_coreRMSD_rounded_drop_dup) # 69826 for Uw

## 1.6 crylig_compound
crylig_compound_df = Uw_for_SAR_info_df[['Cry_lig_name', 'Similar_compnd_name', 'Similarity', 'Core_num', 'Diff_an', 'Part_fix', 'MCS_smarts']]
crylig_compound_df_simi_rounded = crylig_compound_df.round({'Similarity':2})
crylig_compound_df_simi_rounded_drop_dup = crylig_compound_df_simi_rounded.drop_duplicates().copy()
crylig_compound_df_simi_rounded_drop_dup.rename(columns={'Cry_lig_name':'pdbid', 'Similar_compnd_name':'compound', 'Similarity':'similarity', 'Core_num':'core_heavy_atom_number', 'Diff_an': 'different_heavy_atom_number', 'Part_fix':'fix_part_core', 'MCS_smarts':'core_smarts'}, inplace=True) #'different_heavy_atom_number':计算原子数最**多**的小分子的原子数与MCS数的差异; 'Part_fix': no 'completeRingsOnly' - 'completeRingsOnly' > 6
crylig_compound_df_simi_rounded_drop_dup.to_csv(f'{wrkdir}/planet_Uw/crylig_compound.csv', sep='\t', index=False)
# len(crylig_compound_df_simi_rounded_drop_dup) #69400

## 1.7 pdbid_mapped_uniprot_family_chembl_info
### target_info_from_chembl
CHEMBL_map_to_Uniprot = pd.read_csv(f'{wrkdir}/ChEMBLid_to_uniprot_id_chembl30.csv', sep='\t', names=['uniprot_id_from_chembl', 'target_chembl_id', 'target_name_from_chembl', 'target_type_from_chembl'], header=None, skiprows = 1)
CHEMBL_name_type = CHEMBL_map_to_Uniprot[['target_chembl_id', 'target_name_from_chembl', 'target_type_from_chembl']].copy()
CHEMBL_name_type.drop_duplicates(inplace=True)
joined_df = CHEMBL_map_to_Uniprot.groupby('target_chembl_id')['uniprot_id_from_chembl'].agg(','.join).reset_index()
joined_uniprot_name_type = pd.merge(joined_df, CHEMBL_name_type, on='target_chembl_id')
Uw_chembl_info = joined_uniprot_name_type[joined_uniprot_name_type['target_chembl_id'].isin(list(set(Uw_for_SAR_info_df['Target_chembl_id'])))]
# Uw_chembl_info.to_csv(f'{wrkdir}/planet_Uw/target_info_from_chembl.csv', sep='\t', index=False)

### pdbid_to_uniprot
uniprot_mapped_df = pd.read_csv(f'{wrkdir}/converted_PDBIDs_INDEX_general_PL_data.2019.tab.tsv', sep='\t')
uniprot_mapped_df['PDB ID']=uniprot_mapped_df['PDB ID'].map(lambda x:str(x).split(','))
uniprot_mapped_df_exploded=uniprot_mapped_df.explode('PDB ID') # 14025, 一个pdbid(共14025-12787/14025个)对应多个Uniprot ID
uniprot_mapped_df_exploded_simpled_pdb = uniprot_mapped_df_exploded[['PDB ID', 'Entry']].copy()
uniprot_mapped_df_exploded_simpled_pdb.rename(columns={'PDB ID':'pdbid', 'Entry': 'uniprot_id_for_pdbid'}, inplace=True)
pdbid_unip = uniprot_mapped_df_exploded_simpled_pdb.groupby('pdbid')['uniprot_id_for_pdbid'].agg(','.join).reset_index()
Uw_pdbid_uniprot = pdbid_unip[pdbid_unip['pdbid'].isin(list(Uw_for_SAR_info_df['Cry_lig_name']))]
# Uw_pdbid_uniprot.to_csv(f'{wrkdir}/planet_Uw/pdbid_to_uniprot_id.csv', sep='\t', index=False)

### pdbid_mapped_chembl_intersected_uniprot
target_to_pdbid = Uw_for_SAR_info_df[['Target_chembl_id', 'Cry_lig_name']].drop_duplicates()
target_to_pdbid.rename(columns={'Target_chembl_id':'target_chembl_id', 'Cry_lig_name':'pdbid'}, inplace=True)
# Uw_chembl_info = pd.read_csv(f'{wrkdir}//planet_Uw/target_info_from_chembl.csv', sep='\t') #803
Uw_chembl_info['uniprot_id']=Uw_chembl_info['uniprot_id_from_chembl'].map(lambda x:x.split(','))
Uw_chembl_info_exploded = Uw_chembl_info.explode('uniprot_id') #880
# Uw_pdbid_uniprot = pd.read_csv(f'{wrkdir}/planet_Uw/pdbid_to_uniprot_id.csv', sep='\t') #5908
Uw_pdbid_uniprot['uniprot_id']=Uw_pdbid_uniprot['uniprot_id_for_pdbid'].map(lambda x:x.split(','))
Uw_pdbid_uniprot_exploded = Uw_pdbid_uniprot.explode('uniprot_id') #6274
target_pdbid_uniprot_mapped = pd.merge(
    pd.merge(target_to_pdbid, Uw_chembl_info_exploded, on='target_chembl_id'), 
    Uw_pdbid_uniprot_exploded, 
    on=['pdbid', 'uniprot_id'])[['pdbid', 'uniprot_id_for_pdbid', 'uniprot_id', 'target_chembl_id', 'uniprot_id_from_chembl', 'target_name_from_chembl', 'target_type_from_chembl']]

### from pharos
pharos_mapped_df = pd.read_csv(f'{wrkdir}/pharos_query_20220215.csv', sep=',')
pharos_mapped_df.drop(columns='id', inplace=True)
pharos_mapped_df.rename(columns={'UniProt': 'uniprot_id', 'Name': 'target_name_from_pharos', 'Family': 'target_family_from_pharos'}, inplace=True)
target_pdbid_uniprot_mapped_pharos = pd.merge(target_pdbid_uniprot_mapped, pharos_mapped_df, on='uniprot_id', how='left')
# len(target_pdbid_uniprot_mapped_pharos[target_pdbid_uniprot_mapped_pharos['target_family_from_pharos'].isna()]) #826
target_pdbid_uniprot_mapped_pharos.loc[target_pdbid_uniprot_mapped_pharos['target_family_from_pharos'].isna(), 'target_family_from_pharos'] = 'Unclassified'
target_pdbid_uniprot_mapped_pharos.loc[target_pdbid_uniprot_mapped_pharos['target_name_from_pharos'].isna(), 'target_name_from_pharos'] = 'None'

uniprot_mapped_df = pd.read_csv(f'{wrkdir}/converted_PDBIDs_INDEX_general_PL_data.2019.tab.tsv', sep='\t')
uniprot_name_df = uniprot_mapped_df[['Entry', 'Protein names']].copy()
uniprot_name_df.rename(columns={'Entry':'uniprot_id', 'Protein names':'target_name_from_uniprot'}, inplace=True)
target_pdbid_uniprot_mapped_pharos_uniprot_name = pd.merge(target_pdbid_uniprot_mapped_pharos, uniprot_name_df, on='uniprot_id')
target_pdbid_uniprot_mapped_pharos_uniprot_name = target_pdbid_uniprot_mapped_pharos_uniprot_name[['pdbid', 'uniprot_id_for_pdbid', 'uniprot_id', 'target_name_from_uniprot', 'target_name_from_pharos', 'target_family_from_pharos', 'target_chembl_id', 'uniprot_id_from_chembl', 'target_name_from_chembl', 'target_type_from_chembl']].drop_duplicates()
target_pdbid_uniprot_mapped_pharos_uniprot_name.to_csv(f'{wrkdir}/planet_Uw/pdbid_mapped_uniprot_family_chembl_info.csv', sep='\t', index=False)
