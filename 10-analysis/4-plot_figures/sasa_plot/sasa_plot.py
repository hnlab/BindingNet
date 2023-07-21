'''
pAffi_SASA in PDBbind_subset and BindingNet
'''
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statannotations.Annotator import Annotator

index_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PIP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset.csv', sep='\t')
PLANet_df = pd.read_csv(f'{index_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv', sep='\t')
PIPUP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset_union_BindingNet.csv', sep='\t')
# print(len(PIP_df), len(PLANet_df), len(PIPUP_df))

all_prop_sasa_pfam = pd.read_csv('../../3-property_calculate/2-other_property/all_prop_sasa_pfam.csv', sep='\t')
# len(all_prop_sasa_pfam)

PIP_prop = all_prop_sasa_pfam[all_prop_sasa_pfam['unique_identify'].isin(PIP_df['pdb_id'])].copy()
PLANet_prop = all_prop_sasa_pfam[all_prop_sasa_pfam['unique_identify'].isin(PLANet_df['unique_identify'])].copy()
PIPUP_prop = all_prop_sasa_pfam[all_prop_sasa_pfam['unique_identify'].isin(PIPUP_df['unique_identify'])].copy()

# wrkdir='/pubhome/xli02/project/PLIM/analysis/20230524_bindingnet/sasa_plot'

# BindingNet
if not Path('bindingnet').exists():
    Path('bindingnet').mkdir()
Uw_grouped = PLANet_prop.groupby('PCV_cluster')
clu2pear_SA = {}
for name, gp_df in Uw_grouped:
    if len(gp_df) > 20:
        x_ = gp_df['del_sasa']
        y_ = gp_df['-logAffi']
        pear = stats.pearsonr(x_, y_)
        spear = stats.spearmanr(x_, y_)
        clu2pear_SA[name] = round(pear[0], 3)

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.regplot(
            x=x_, y=y_,
            data=gp_df,
            ax=ax
        )
        ax.set_title(f'buried SASA-pAffi for {name} in BindingNet (N={len(gp_df)})', fontsize=15)
        ax.set_ylabel('Experimental pAffi', fontsize=15)
        ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
        ax.text(0.5,0.99,
            f'Rp:{float(pear[0]):.3f}_{float(pear[1]):.3f}; Rs:{float(spear[0]):.3f}_{float(spear[1]):.3f}',
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            zorder=3, fontsize=13)
        plt.savefig(f'bindingnet/{name}.png', dpi=600, bbox_inches='tight')
        plt.close()

Uw_PCV_SASA_pAffi_Rp_df = pd.DataFrame.from_dict(clu2pear_SA, orient='index', columns=['Rp(SASA,pAffi)']).reset_index()
Uw_PCV_SASA_pAffi_Rp_df.rename(columns={'index':'PCV_cluster'}, inplace=True)

# PDBbind_subset
if not Path('PDBbind_subset').exists():
    Path('PDBbind_subset').mkdir()
PIP_grouped = PIP_prop.groupby('PCV_cluster')
clu2pear_SA = {}
for name, gp_df in PIP_grouped:
    if len(gp_df) > 20:
        x_ = gp_df['del_sasa']
        y_ = gp_df['-logAffi']
        pear = stats.pearsonr(x_, y_)
        spear = stats.spearmanr(x_, y_)
        clu2pear_SA[name] = round(pear[0], 3)

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.regplot(
            x=x_, y=y_,
            data=gp_df,
            ax=ax
        )
        ax.set_title(f'buried SASA-pAffi for {name} in PDBbind_subset (N={len(gp_df)})', fontsize=15)
        ax.set_ylabel('Experimental pAffi', fontsize=15)
        ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
        ax.text(0.5,0.99,
            f'Rp:{float(pear[0]):.3f}_{float(pear[1]):.3f}; Rs:{float(spear[0]):.3f}_{float(spear[1]):.3f}',
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            zorder=3, fontsize=13)
        plt.savefig(f'PDBbind_subset/{name}.png', dpi=600, bbox_inches='tight')
        plt.close()

PIP_PCV_SASA_pAffi_Rp_df = pd.DataFrame.from_dict(clu2pear_SA, orient='index', columns=['Rp(SASA,pAffi)']).reset_index()
PIP_PCV_SASA_pAffi_Rp_df.rename(columns={'index':'PCV_cluster'}, inplace=True)

# PIPUP
if not Path('PIPUP').exists():
    Path('PIPUP').mkdir()
PIPUP_grouped = PIPUP_prop.groupby('PCV_cluster')
clu2pear_SA = {}
for name, gp_df in PIPUP_grouped:
    if len(gp_df) > 20:
        x_ = gp_df['del_sasa']
        y_ = gp_df['-logAffi']
        pear = stats.pearsonr(x_, y_)
        spear = stats.spearmanr(x_, y_)
        clu2pear_SA[name] = round(pear[0], 3)

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.regplot(
            x=x_, y=y_,
            data=gp_df,
            ax=ax
        )
        ax.set_title(f'buried SASA-pAffi for {name} in PDBbind_subset' + r'$\cup$' f'BindingNet (N={len(gp_df)})', fontsize=15)
        ax.set_ylabel('Experimental pAffi', fontsize=15)
        ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
        ax.text(0.5,0.99,
            f'Rp:{float(pear[0]):.3f}_{float(pear[1]):.3f}; Rs:{float(spear[0]):.3f}_{float(spear[1]):.3f}',
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            zorder=3, fontsize=13)
        plt.savefig(f'PIPUP/{name}.png', dpi=600, bbox_inches='tight')
        plt.close()
PIPUP_PCV_SASA_pAffi_Rp_df = pd.DataFrame.from_dict(clu2pear_SA, orient='index', columns=['Rp(SASA,pAffi)']).reset_index()
PIPUP_PCV_SASA_pAffi_Rp_df.rename(columns={'index':'PCV_cluster'}, inplace=True)

Uw_PCV_SASA_pAffi_Rp_df['dataset'] = 'BindingNet'
PIP_PCV_SASA_pAffi_Rp_df['dataset'] = 'PDBbind_subset'
PIPUP_PCV_SASA_pAffi_Rp_df['dataset'] = 'PIPUP'
concated_df = pd.concat([PIP_PCV_SASA_pAffi_Rp_df, Uw_PCV_SASA_pAffi_Rp_df, PIPUP_PCV_SASA_pAffi_Rp_df])

grouped_median = concated_df.groupby(['dataset']).median().reset_index()

order = ['PDBbind_subset', 'BindingNet', 'PIPUP']
fig, ax= plt.subplots(figsize=(12,8))
# fig.autofmt_xdate()
sns.boxplot(x="dataset", y="Rp(SASA,pAffi)", data=concated_df, linewidth=2.5, order = order)
sns.swarmplot(x = "dataset", y = "Rp(SASA,pAffi)", data = concated_df, dodge=True, color="black", alpha = 0.5, size = 6, order = order)
ax.set_title(f'Clustered Rp(SASA, pAffi)', fontsize=20)
ax.set_xlabel(f'dataset', fontsize=20)
ax.set_ylabel(f'Rp(SASA,pAffi)', fontsize=20)
ax.tick_params(axis='x', labelsize= 15)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

vertical_offset = concated_df['Rp(SASA,pAffi)'].median() * 0.03 # offset from median for display
for i,dataset_ in enumerate(order):
    median_metric = round(grouped_median.loc[grouped_median['dataset'] == dataset_]['Rp(SASA,pAffi)'].values[0], 3)
    ax.text(i, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold', c='w', fontsize='x-large')

pairs = [("BindingNet", "PDBbind_subset"), 
    ('PIPUP', "PDBbind_subset"), 
    ]
annotator = Annotator(ax, pairs, x="dataset", y="Rp(SASA,pAffi)", data=concated_df, order = order)
annotator.configure(test='t-test_ind').apply_and_annotate()

plt.savefig(f'PCV_clustered_Rp_SA_for_3_dataset_box.png', dpi=300, bbox_inches='tight')
plt.close()

Uw_PCV_SASA_pAffi_Rp_df['dataset'] = 'BindingNet'
PIP_PCV_SASA_pAffi_Rp_df['dataset'] = 'PDBbind_subset'
PIPUP_PCV_SASA_pAffi_Rp_df['dataset'] = 'PIPUP'
merged_df = pd.merge(PIP_PCV_SASA_pAffi_Rp_df.drop(columns='dataset').rename(columns={'Rp(SASA,pAffi)':'PDBbind_subset'}), Uw_PCV_SASA_pAffi_Rp_df.drop(columns='dataset').rename(columns={'Rp(SASA,pAffi)':'BindingNet'}), on='PCV_cluster') # 36
merged_melted_df = merged_df.sort_values(by='PDBbind_subset').melt(id_vars=['PCV_cluster'], value_vars=['PDBbind_subset', 'BindingNet'], var_name='dataset', value_name='Rp(SASA,pAffi)')

fig, ax = plt.subplots(figsize=(40, 7))
fig.autofmt_xdate()
sns.barplot(x="PCV_cluster", y="Rp(SASA,pAffi)", hue="dataset", data=merged_melted_df, palette={'PDBbind_subset': 'royalblue', 'BindingNet':'coral'})
# sns.barplot(x="PCV_cluster", y="Rp(SASA,pAffi)", hue="dataset", data=merged_melted_df)
ax.set_title(f'Clustered Rp(SASA, pAffi) for PDBbind_subset and BindingNet', fontsize=30)
ax.set_xlabel(f'Pfam clusters', fontsize=20)
ax.set_ylabel(f'Rp(SASA,pAffi)', fontsize=20)
ax.tick_params(axis='x', labelsize= 20)
ax.tick_params(axis='y', labelsize= 20)
plt.setp(ax.get_legend().get_texts(), fontsize='20')
plt.setp(ax.get_legend().get_title(), fontsize='20')
plt.savefig(f'PCV_clustered_Rp_SA_PDBbind_subset_vs_BindingNet.png', dpi=300, bbox_inches='tight')
plt.close()
