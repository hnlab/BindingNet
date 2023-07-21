from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

index_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PIP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset.csv', sep='\t')
PLANet_df = pd.read_csv(f'{index_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv', sep='\t')
PIPUP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset_union_BindingNet.csv', sep='\t')
# print(len(PIP_df), len(PLANet_df), len(PIPUP_df))


# 1. property calculation
PDBbind_whole = pd.read_csv('PDBbind_dealt.csv', sep='\t')
skipped_mols = []
for row in PDBbind_whole.itertuples():
    uniq_id = row.pdb_id
    # lig_file = f'{pdbbind_dir}/general_structure_only/{uniq_id}/{uniq_id}_ligand.smi'
    lig_file = f'/pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019/{uniq_id}/cry_lig_opt_converted.sdf'
    if not Path(lig_file).exists():
        print(f'{uniq_id} of PDBbind not exsits, skipped.')
        skipped_mols.append(uniq_id)
        continue
    compnd_mol = Chem.SDMolSupplier(lig_file)[0]
    if compnd_mol is None:
        print(f'For compounds in PDBbind, {uniq_id} cannot be read by rdkit, skipped.')
        skipped_mols.append(uniq_id)
        continue
    PDBbind_whole.loc[row.Index, 'mw'] = Descriptors.MolWt(compnd_mol)
    PDBbind_whole.loc[row.Index, 'logp'] = Descriptors.MolLogP(compnd_mol)
    PDBbind_whole.loc[row.Index, 'rotb'] = Descriptors.NumRotatableBonds(compnd_mol)
    PDBbind_whole.loc[row.Index, 'hbd'] = Descriptors.NumHDonors(compnd_mol)
    PDBbind_whole.loc[row.Index, 'hba'] = Descriptors.NumHAcceptors(compnd_mol)
    PDBbind_whole.loc[row.Index, 'q'] = Chem.GetFormalCharge(compnd_mol)
    PDBbind_whole.loc[row.Index, 'HA'] = compnd_mol.GetNumAtoms()
PDBbind_whole.to_csv('PDBbind_whole.csv', sep='\t', index=False)

PLANet_skipped_mols = []
for row in PLANet_df.itertuples():
    lig_file = f'/pubhome/xli02/project/PLIM/v2019_dataset/web_client_{row.Target_chembl_id}/{row.Target_chembl_id}_{row.Cry_lig_name}/{row.Similar_compnd_name}/compound.sdf'
    if not Path(lig_file).exists():
        print(f'{uniq_id} of PLANet not exsits, skipped.')
        skipped_mols.append(uniq_id)
        continue
    compnd_mol = Chem.SDMolSupplier(lig_file)[0]
    if compnd_mol is None:
        print(f'For compounds in PLANet, {uniq_id} cannot be read by rdkit, skipped.')
        skipped_mols.append(uniq_id)
        continue
    PLANet_df.loc[row.Index, 'mw'] = Descriptors.MolWt(compnd_mol)
    PLANet_df.loc[row.Index, 'logp'] = Descriptors.MolLogP(compnd_mol)
    PLANet_df.loc[row.Index, 'rotb'] = Descriptors.NumRotatableBonds(compnd_mol)
    PLANet_df.loc[row.Index, 'hbd'] = Descriptors.NumHDonors(compnd_mol)
    PLANet_df.loc[row.Index, 'hba'] = Descriptors.NumHAcceptors(compnd_mol)
    PLANet_df.loc[row.Index, 'q'] = Chem.GetFormalCharge(compnd_mol)
    PLANet_df.loc[row.Index, 'HA'] = compnd_mol.GetNumAtoms()
# PLANet_df.to_csv('PLANet_property.csv', sep='\t', index=False)

PIP_prop = PDBbind_whole[PDBbind_whole['pdb_id'].isin(PIP_df['pdb_id'])].copy()
# PIP_prop.to_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PIP_prop.csv', sep='\t', index=False)

PIPUP_prop_df = pd.concat([PIP_prop.rename(columns={'pdb_id':'unique_identify'}), PLANet_df[['unique_identify', '-logAffi', 'mw', 'logp', 'rotb', 'hbd', 'hba', 'q', 'HA']]])
# PIPUP_prop_df.to_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PIPUP_prop.csv', sep='\t', index=False)


# 2. add sasa info
# PDBbind_whole_property = pd.read_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PDBbind_whole.csv', sep='\t')
PDBbind_whole_sasa = pd.read_csv('../1-sasa/PDBbind_whole_sasa.csv')
PDBbind_whole_prop_sasa = pd.merge(PDBbind_whole, PDBbind_whole_sasa.rename(columns={'unique_identity':'pdb_id'}), on=['pdb_id'])

# PLANet_property = pd.read_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PLANet_property.csv', sep='\t')
PLANet_whole_sasa = pd.read_csv('../1-sasa/PLANet_property_sasa.csv')


# 3. add pfam info(PCV_cluster)
PDBbind_v2020_cluster = pd.read_csv('PDBbind_2020_general_refine_classified.csv')
PDBbind_v2020_cluster_sel = PDBbind_v2020_cluster[['pdb', 'PCV_cluster']].copy()
PDBbind_v2020_cluster_sel.rename(columns={'pdb':'pdb_id'}, inplace=True)
PDBbind_whole_prop_sasa_pfam = pd.merge(PDBbind_whole_prop_sasa, PDBbind_v2020_cluster_sel, on='pdb_id')
# PDBbind_whole_prop_sasa_pfam.to_csv('/pubhome/xli02/project/PLIM/analysis/20220829_paper/distribution/property/PDBbind_property_sasa_pfam.csv', sep='\t', index=False)

PLANet_whole_prop_sasa_pfam = pd.merge(PLANet_df.merge(PLANet_whole_sasa.rename(columns={'unique_identity': 'unique_identify'}), on='unique_identify'), PDBbind_v2020_cluster_sel.rename(columns={'pdb_id':'Cry_lig_name'}), on='Cry_lig_name')
# PLANet_whole_prop_sasa_pfam[['unique_identify', '-logAffi', 'mw', 'logp', 'rotb', 'hbd', 'hba', 'q', 'HA', 'lig_sasa', 'rec_sasa', 'com_sasa', 'del_sasa', 'PCV_cluster']].to_csv('/pubhome/xli02/project/PLIM/analysis/20220829_paper/distribution/property/PLANet_whole_prop_sasa_pfam.csv', sep='\t', index=False)

all_prop_sasa_pfam = pd.concat([PDBbind_whole_prop_sasa_pfam.rename(columns={'pdb_id':'unique_identify'}), PLANet_whole_prop_sasa_pfam[['unique_identify', '-logAffi', 'mw', 'logp', 'rotb', 'hbd', 'hba', 'q', 'HA', 'lig_sasa', 'rec_sasa', 'com_sasa', 'del_sasa', 'PCV_cluster']]])
all_prop_sasa_pfam.to_csv('all_prop_sasa_pfam.csv', sep='\t', index=False)


# plot: /pubhome/xli02/project/PLIM/analysis/20220812_paper/20220812_distribution_of_PIP_and_PLANet.ipynb
# sns.kdeplot(PIP_prop['-logAffi'])
# sns.kdeplot(PLANet_df['-logAffi'])
# sns.kdeplot(PIPUP_prop_df['-logAffi'])
# plt.xlabel("Experimental pAffi")
# plt.title(f'The distribution of Experimental pAffi for different datasets')
# plt.legend(labels=[f'PDBbind_subset (N={len(PIP_prop)})', f'PLANet (N={len(PLANet_df)})','PDBbind_subset' +r'$\cup$' f'PLANet (N={len(PIPUP_prop_df)})'], title = "Dataset")
# plt.savefig(f'/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PIP_PLANet_PIPUP_pAffi_kde_plot.png', dpi=300, bbox_inches='tight')
# plt.close()

# fig, ax = plt.subplots()
# sns.histplot(all_prop_df, x='-logAffi', hue="dataset", stat="density", common_norm=False, binwidth=0.5, element="step")
# ax.set_xlabel("Experimental pAffi")
# ax.set_title(f'The distribution of Experimental pAffi for different datasets')
# # plt.legend(labels=[f'PDBbind_subset', f'PLANet','PDBbind_subset' +r'$\cup$' f'PLANet'], title = "Dataset", loc='upper right')
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles[:3], [f'PDBbind_subset', f'PLANet','PDBbind_subset' +r'$\cup$' f'PLANet'])
# ax.legend(ax.get_legend().legendHandles, [f'PDBbind_subset', f'PLANet','PDBbind_subset' +r'$\cup$' f'PLANet'], loc='upper left')
# plt.savefig(f'/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PIP_PLANet_PIPUP_pAffi_hist_plot.png', dpi=300, bbox_inches='tight')
# plt.close()
