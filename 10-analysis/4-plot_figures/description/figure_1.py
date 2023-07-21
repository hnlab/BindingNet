import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt

if not Path('figure_1').exists():
    Path('figure_1').mkdir()

# 1. target classify
index_dir = '/pubhome/xli02/project/bindingnet_database/from_chembl_client/index'
PLANet_df = pd.read_csv(f'{index_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv', sep='\t')
target_pdbid_uniprot_mapped_pharos = pd.read_csv('/home/xli/git/Chembl-Scaffold-database/pipeline/7-web_server/1-mk_table/planet_Uw/pdbid_mapped_uniprot_family_chembl_info.csv', sep='\t')
target_pdbid_uniprot_mapped_pharos.rename(columns={'pdbid':'Cry_lig_name', 'target_chembl_id':'Target_chembl_id'}, inplace=True)
PLANet_classify = pd.merge(PLANet_df, target_pdbid_uniprot_mapped_pharos, on=['Target_chembl_id', 'Cry_lig_name'], how='left') #70215
labels = list(set(PLANet_classify['target_family_from_pharos']))
sizes = [len(PLANet_classify[PLANet_classify['target_family_from_pharos']==x]) for x in labels]
dictionary = dict(zip(labels, sizes))
sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
x = np.char.array(list(sorted_dictionary.keys()))
y = np.array(list(sorted_dictionary.values()))
percent = 100.*y/sum(y)
labels_2 = ['{0}  {1:1.2f}%'.format(i,j) for i,j in zip(x, percent)]
colors = ['b', 'c', 'g', 'y', 'r', 'pink', 'darkorange', 'wheat', 'k', 'grey', 'lime']

fig, ax= plt.subplots(figsize=(6,4))
patches, texts = plt.pie(sorted(sizes, reverse=True), startangle=90, colors=colors, counterclock=False, wedgeprops={'edgecolor': 'w', 'linewidth':0.1}, radius=1)
plt.legend(patches, labels_2, loc="best", bbox_to_anchor=(1, 0.9), prop={'size':10})
plt.title(f'Target classification of BindingNet (N={len(PLANet_df)})', y=1, fontsize=15)
plt.savefig('figure_1/PLANet_target_classify.png', dpi=600, bbox_inches='tight')
plt.close()

# 2. compound count for per target
target_count_df = pd.DataFrame({'count':PLANet_df.groupby('Target_chembl_id').size()}).reset_index()
bins=[0, 1, 10, 50, 100, 1000, 3730]
labels=['1', '2-10', '11-50', '51-100', '101-1000', '>1000']
target_count_df['count_group'] = pd.cut(target_count_df['count'], bins=bins, labels=labels)
fig, ax= plt.subplots(figsize=(6,4))
target_count_df['count_group'].value_counts(sort=False).plot.bar(rot=0, color='b')
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.title('Distribution of compounds counts for per target', fontsize=15)
plt.xlabel('Number of compounds / target', fontsize=15)
plt.ylabel('Number of targets', fontsize=15)
plt.savefig('figure_1/PLANet_compounds_count_for_each_target.png', dpi=600, bbox_inches='tight')
plt.close()

# 3. core_RMSD
fig, ax= plt.subplots(figsize=(8,6))
sns.histplot(PLANet_df, x="Core_RMSD", color='b', binwidth=0.02)
ax.set_title(f'The distribution of MCS_RMSD in BindingNet (N={len(PLANet_df)})', fontsize=15)
ax.set_xlabel('MCS_RMSD', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig('figure_1/PLANet_core_RMSD_distribution_Uw_scale.png', dpi=600, bbox_inches='tight')
plt.close()

# 4. total_delta
fig, ax= plt.subplots(figsize=(6,6))
sns.histplot(PLANet_df, x="Total_delta", color='b', binwidth=10)
ax.set_title('The distribution of $\it{E}_{cmx}$' + f' in BindingNet (N={len(PLANet_df)})', fontsize=15)
ax.set_xlabel('$\it{E}_{cmx}$', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig('figure_1/PLANet_Ecmx_distribution_Uw.png', dpi=600, bbox_inches='tight')
plt.close()

# 5. lig_delta
fig, ax= plt.subplots(figsize=(6,6))
sns.histplot(PLANet_df, x="Lig_delta", color='b', binwidth=0.2)
ax.set_title('The distribution of $\it{E}_{dL}$' + f' in BindingNet (N={len(PLANet_df)})', fontsize=15)
ax.set_xlabel('$\it{E}_{dL}$', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig('figure_1/PLANet_lig_delta_distribution_Uw.png', dpi=600, bbox_inches='tight')
plt.close()

# 6. pAffi
# ## 6.1 bindingnet
# fig, ax= plt.subplots(figsize=(6,6))
# sns.histplot(PLANet_df, x="-logAffi", color='b', binwidth=0.2)
# ax.set_title(f'The distribution of pAffi in BindingNet (N={len(PLANet_df)})', fontsize=15)
# ax.set_xlabel('Experimental pAffi', fontsize=15)
# ax.set_ylabel('Count', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# ax.set_xlim(0,16)
# plt.savefig('figure_1/PLANet_Binding_affinity_distribution_Uw.png', dpi=600, bbox_inches='tight')
# plt.close()

# ## 6.2 PDBbind_general
# PDBbind_dir = '/pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019'
# PDBbind_v19_general = f'{PDBbind_dir}/INDEX_general_PL_data_grepped.2019'
# PDBbind_v19_general_df = pd.read_csv(PDBbind_v19_general, sep='\t')

# fig, ax= plt.subplots(figsize=(6,6))
# sns.histplot(PDBbind_v19_general_df, x="-logAffi", color='b', binwidth=0.2)
# ax.set_title(f'The distribution of pAffi in PDBbind_general (N={len(PDBbind_v19_general_df)})', fontsize=15)
# ax.set_xlabel('Experimental pAffi', fontsize=15)
# ax.set_ylabel('Count', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# ax.set_xlim(0,16)
# plt.savefig('figure_1/PDBbind_general/Binding_affinity_distribution_PDBbind_general.png', dpi=600, bbox_inches='tight')
# plt.close()

# ## 6.3 PDBbind_minimized
# PDBbind_minimized_df = pd.read_csv('/pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019/index/PDBbind_v19_minimized_succeed_RMSD_manually_modified.csv', sep='\t')
# fig, ax= plt.subplots(figsize=(6,6))
# sns.histplot(PDBbind_minimized_df, x="-logAffi", color='b', binwidth=0.2)
# ax.set_title(f'The distribution of pAffi in PDBbind_minimized (N={len(PDBbind_minimized_df)})', fontsize=15)
# ax.set_xlabel('Experimental pAffi', fontsize=15)
# ax.set_ylabel('Count', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# ax.set_xlim(0,16)
# plt.savefig('figure_1/PDBbind_minimized/PDBbind_minimized_Binding_affinity_distribution.png', dpi=600, bbox_inches='tight')
# plt.close()

# ## 6.4 PDBbind_subset
PIP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset.csv', sep='\t')
# fig, ax= plt.subplots(figsize=(6,6))
# sns.histplot(PIP_df, x="-logAffi", color='b', binwidth=0.2)
# ax.set_title(f'The distribution of pAffi in PDBbind_subset (N={len(PIP_df)})', fontsize=15)
# ax.set_xlabel('Experimental pAffi', fontsize=15)
# ax.set_ylabel('Count', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# ax.set_xlim(0,16)
# plt.savefig('figure_1/PDBbind_minimized/PDBbind_subset_Binding_affinity_distribution.png', dpi=600, bbox_inches='tight')
# plt.close()

## 6.5 BindingNet vs PDBbind_subset
# fig, ax= plt.subplots(figsize=(8,6))
# sns.kdeplot(PDBbind_v19_general_df['-logAffi'])
# # sns.kdeplot(PDBbind_minimized_df['-logAffi'])
# # sns.kdeplot(PIP_df['-logAffi'])
# sns.kdeplot(PLANet_df['-logAffi'])
# ax.set_xlabel('Experimental pAffi', fontsize=15)
# ax.set_ylabel('Density', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# plt.xlabel("Experimental pAffi")
# ax.set_title(f'The distribution of Experimental pAffi for PDBbind_general and BindingNet', fontsize=15)
# plt.legend(labels=[f'PDBbind_general (N={len(PDBbind_v19_general_df)})', f'BindingNet (N={len(PLANet_df)})'], title = "dataset")
# plt.setp(ax.get_legend().get_texts(), fontsize='12')
# plt.setp(ax.get_legend().get_title(), fontsize='12')
# plt.savefig(f'figure_1/PDBbind_general_PLANet_dataset_pAffi_kde_plot.png', dpi=300, bbox_inches='tight')
# plt.close()

# ## 6.6 BindingNet + PDBbind_general + PDBbind_minimized + PDBbind_subset
# fig, ax= plt.subplots(figsize=(8,6))
# sns.kdeplot(PDBbind_v19_general_df['-logAffi'])
# sns.kdeplot(PDBbind_minimized_df['-logAffi'])
# sns.kdeplot(PIP_df['-logAffi'])
# sns.kdeplot(PLANet_df['-logAffi'])
# plt.xlabel("Experimental pAffi")
# ax.set_title(f'The distribution of Experimental pAffi for different datasets', fontsize=15)
# plt.legend(labels=[f'PDBbind_general (N={len(PDBbind_v19_general_df)})', f'PDBbind_minimized (N={len(PDBbind_minimized_df)})', f'PDBbind_subset (N={len(PIP_df)})',f'BindingNet (N={len(PLANet_df)})'], title = "dataset")
# plt.setp(ax.get_legend().get_texts(), fontsize='12')
# plt.setp(ax.get_legend().get_title(), fontsize='12')
# ax.set_xlabel('Experimental pAffi', fontsize=15)
# ax.set_ylabel('Density', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# plt.savefig(f'figure_1/four_dataset_pAffi_kde_plot.png', dpi=300, bbox_inches='tight')
# # plt.close()

## 6.7 BindingNet + PDBbind_subset + PDBbind_subset并BindingNet
PIPUP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset_union_BindingNet.csv', sep='\t')
fig, ax= plt.subplots(figsize=(8,6))
# sns.kdeplot(PDBbind_v19_general_df['-logAffi'])
# sns.kdeplot(PDBbind_minimized_df['-logAffi'])
sns.kdeplot(PIP_df['-logAffi'])
sns.kdeplot(PLANet_df['-logAffi'])
sns.kdeplot(PIPUP_df['-logAffi'])
plt.xlabel("Experimental pAffi")
ax.set_title(f'The distribution of Experimental pAffi for different datasets', fontsize=15)
plt.legend(labels=[f'PDBbind_subset (N={len(PIP_df)})',f'BindingNet (N={len(PLANet_df)})', 'PDBbind_subset'+r'$\cup{}$' + f'BindingNet (N={len(PIPUP_df)})'], title = "dataset", loc='upper left')
plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Experimental pAffi', fontsize=15)
ax.set_ylabel('Density', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig(f'figure_1/PIP_PLANet_PIPUP_dataset_pAffi_kde_plot.png', dpi=300, bbox_inches='tight')
plt.close()
