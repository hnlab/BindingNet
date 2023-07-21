from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sum_df=pd.read_csv('output/sum.csv', sep='\t')
selected_df = sum_df[sum_df['dataset'].isin(['PIP', 'PLANet', 'PIPUP']) & sum_df['test_type'].isin(['valid', 'train', 'test', 'CASF_v16_minimized', 'CASF_v16_intersected_Uw_minimized'])].copy()
grouped_median = selected_df.groupby(['test_type', 'dataset', 'model_type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_cmx' if row.model_type == 'complex' else f'{row.dataset}_lig_alone' for row in grouped_median.itertuples()]
grouped = selected_df.groupby('test_type')

test_type_to_test_title={'valid':'validation', 'train':'training', 'test':'self_testing', 'CASF_v16_minimized':'CASF_v16', 'CASF_v16_intersected_Uw_minimized':'CASF_v16_intersected_PLANet'}
order = ['PIP', 'PLANet', 'PIPUP']


# 1. Rp
metric = 'pearsonr'
out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

# test_type = 'CASF_v16_intersected_Uw_minimized'
# test_df = grouped.get_group(test_type)
# fig, ax= plt.subplots(figsize=(7,4))
# sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
# sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
# ax.set_title(f'Rp on CASF_subset(N=115)', fontsize=15)
# vertical_offset = test_df[metric].median() * 0.07 # offset from median for display
# # fig.autofmt_xdate()
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2], loc='lower right')
# for i, modl in enumerate(order):
#     for tp in ['cmx', 'lig_alone']:
#         median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
#         if tp == 'cmx':
#             ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
#         else:
#             ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
# # if metric == 'pearsonr':
# ax.set_ylim(0.4,1)
# ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])
# plt.setp(ax.get_legend().get_texts(), fontsize='12')
# plt.setp(ax.get_legend().get_title(), fontsize='12')
# ax.set_xlabel('Dataset', fontsize=15)
# ax.set_ylabel('Rp', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
# plt.close()


# mannwhitneyu annotated
test_type = 'CASF_v16_intersected_Uw_minimized'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on CASF_subset(N=115)', fontsize=15)
vertical_offset = test_df[metric].median() * 0.07 # offset from median for display

pairs = [(("PLANet", "ligand_alone"), ("PIP", "ligand_alone")), 
    (("PLANet", "complex"), ("PIP", "complex")), 
    ]
annotator = Annotator(ax, pairs, x="dataset", y=metric, data=test_df, hue="model_type", order = order)
annotator.configure(test='t-test_ind').apply_and_annotate()

# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
# if metric == 'pearsonr':
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])
plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15)
ax.set_ylabel('Rp', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set_with_sig.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Rs
metric = 'spearmanr'
out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

test_type = 'CASF_v16_intersected_Uw_minimized'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on CASF_subset(N=115)', fontsize=15)
vertical_offset = test_df[metric].median() * 0.12 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])
plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15)
ax.set_ylabel('Rs', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. RMSE
metric = 'rmse'
out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

test_type = 'CASF_v16_intersected_Uw_minimized'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on CASF_subset(N=115)', fontsize=15)
vertical_offset = test_df[metric].median() * 0.2 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4, 2)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])
plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15)
ax.set_ylabel('RMSE', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. MAE
metric = 'mae'
out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

# test_type = 'CASF_v16_intersected_Uw_minimized'
# test_df = grouped.get_group(test_type)
# fig, ax= plt.subplots(figsize=(7,4))
# sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
# sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
# ax.set_title(f'MAE on CASF_subset(N=115)', fontsize=15)
# vertical_offset = test_df[metric].median() * 0.2 # offset from median for display
# # fig.autofmt_xdate()
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2])
# for i, modl in enumerate(order):
#     for tp in ['cmx', 'lig_alone']:
#         median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
#         if tp == 'cmx':
#             ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
#         else:
#             ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
#     # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
#     ax.set_ylim(0.3, 1.5)
# ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])
# plt.setp(ax.get_legend().get_texts(), fontsize='12')
# plt.setp(ax.get_legend().get_title(), fontsize='12')
# ax.set_xlabel('Dataset', fontsize=15)
# ax.set_ylabel('MAE', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
# plt.close()

test_type = 'CASF_v16_intersected_Uw_minimized'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on CASF_subset(N=115)', fontsize=15)
vertical_offset = test_df[metric].median() * 0.2 # offset from median for display

pairs = [(("PLANet", "ligand_alone"), ("PIP", "ligand_alone")), 
    (("PLANet", "complex"), ("PIP", "complex")), 
    ]
annotator = Annotator(ax, pairs, x="dataset", y=metric, data=test_df, hue="model_type", order = order)
annotator.configure(test='t-test_ind').apply_and_annotate()

# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.3, 1.5)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])
plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15)
ax.set_ylabel('MAE', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set_with_sig.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Fig C: BindingNet_cmx vs PDBbind_subset_cmx
CIP_cmx=pd.read_csv('output/Core_inter_Uw_scatter_PLANet_vs_PIP_cmx_mean.csv', sep='\t')

PLANet_rmse = np.sqrt(mean_squared_error(y_true=CIP_cmx['y_true'], y_pred=CIP_cmx['PLANet_cmx_mean']))
PLANet_pearsonr = stats.pearsonr(CIP_cmx['y_true'], CIP_cmx['PLANet_cmx_mean'])
PLANet_mae = mean_absolute_error(y_true=CIP_cmx['y_true'], y_pred=CIP_cmx['PLANet_cmx_mean'])
PLANet_spearmanr = stats.spearmanr(CIP_cmx['y_true'], CIP_cmx['PLANet_cmx_mean'])


PIP_rmse = np.sqrt(mean_squared_error(y_true=CIP_cmx['y_true'], y_pred=CIP_cmx['PIP_cmx_mean']))
PIP_pearsonr = stats.pearsonr(CIP_cmx['y_true'], CIP_cmx['PIP_cmx_mean'])
PIP_mae = mean_absolute_error(y_true=CIP_cmx['y_true'], y_pred=CIP_cmx['PIP_cmx_mean'])
PIP_spearmanr = stats.spearmanr(CIP_cmx['y_true'], CIP_cmx['PIP_cmx_mean'])
print(f'PLANet Rp:{PLANet_pearsonr[0]}, MAE:{PLANet_mae}, Rs:{PLANet_spearmanr[0]}, RMSE:{PLANet_rmse}')
print(f'PIP Rp:{PIP_pearsonr[0]}, MAE:{PIP_mae}, Rs:{PIP_spearmanr[0]}, RMSE:{PIP_rmse}')

fig, ax = plt.subplots(figsize=(6,6))
sns.regplot(data=CIP_cmx, x="y_true", y="PIP_cmx_mean")
sns.regplot(data=CIP_cmx, x="y_true", y="PLANet_cmx_mean")
# sns.regplot(data=delta_df, x="y_true", y="PLANet_cmx_mean", marker="o", facecolor="none", edgecolor='black', linewidth=2)
# sns.scatterplot(data=delta_df, x="y_true", y="PIP_cmx_mean", marker="o", facecolor="none", edgecolor='dimgrey', linewidth=2)
lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
# plt.xlabel("Experimental pAffi")
# plt.ylabel("Mean predicted pAffi")
ax.set_title('CASF_subset (N=115)',fontsize=15)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2], loc='upper left')
plt.legend(labels=[f"PDBbind_subset_cmx_mean: Rp_{round(PIP_pearsonr[0],3)}_MAE_{round(PIP_mae,3)}", f"BindingNet_cmx_mean: Rp_{round(PLANet_pearsonr[0],3)}_MAE_{round(PLANet_mae,3)}"], loc='upper left')
plt.setp(ax.get_legend().get_texts(), fontsize='11')
plt.setp(ax.get_legend().get_title(), fontsize='11')
ax.set_xlabel('Experimental pAffi', fontsize=15)
ax.set_ylabel('Mean predicted pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig(f'Core_inter_Uw_scatter_PLANet_vs_PIP_cmx_mean_MAE_regplot_rename.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Fig D: similarity
max_simi_distribution = pd.read_csv('similarity/all_models_max_simi_distribution.csv', sep='\t')
fig, ax= plt.subplots(figsize=(8,6))
sns.kdeplot(max_simi_distribution['PDBbind_minimized_intersected_Uw'])
sns.kdeplot(max_simi_distribution['PLANet_Uw'])
sns.kdeplot(max_simi_distribution['PDBbind_minimized_intersected_Uw_union_Uw'])
# plt.xlabel("Tanimoto similarity")
plt.xlim(0,1)
ax.set_title(f'Best similarity distribution among CASF_subset and dataset', fontsize=15)
plt.legend(labels=['PDBbind_subset','BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'], title = "Dataset")

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Tanimoto similarity', fontsize=15) #
ax.set_ylabel('Density', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'similarity/PDBbind_intersected_Uw_vs_Uw_on_CIP.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. SASA Fig 9D: Mw, Heavy atom number, SASA
from scipy.stats import gaussian_kde

wrkdir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim'
core_intersected_Uw_df = pd.read_csv('index/core_intersected_Uw.csv', sep='\t')
all_prop_sasa_pfam = pd.read_csv('../../../3-property_calculate/2-other_property/all_prop_sasa_pfam.csv', sep='\t')
CIP_prop_df_sasa = all_prop_sasa_pfam[all_prop_sasa_pfam['unique_identify'].isin(core_intersected_Uw_df['pdb_id'])].drop(columns=['PCV_cluster']).rename(columns={'unique_identify': 'pdb_id'})

# CIP_prop_df_sasa = pd.read_csv('/pubhome/xli02/project/PLIM/analysis/20220829_paper/ML/rm_core_all_simi_1/property/property.csv', sep='\t')
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)

## 7.1 Mw
pearsonr = round(stats.pearsonr(CIP_prop_df_sasa['mw'], CIP_prop_df_sasa['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(CIP_prop_df_sasa['mw'], CIP_prop_df_sasa['-logAffi'])[0],3)

xy = np.vstack([CIP_prop_df_sasa['mw'], CIP_prop_df_sasa['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = CIP_prop_df_sasa['mw'][idx], CIP_prop_df_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('Molecular weight (Da)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,1000)
ax.set_ylim(0,16)
ax.set_title(f'CASF_subset (N={len(CIP_prop_df_sasa)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('property/CIP_Mw_pAffi_density_scaled_1000.png', dpi=300, bbox_inches='tight')
plt.close()


## 7.3 HAN
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(CIP_prop_df_sasa['HA'], CIP_prop_df_sasa['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(CIP_prop_df_sasa['HA'], CIP_prop_df_sasa['-logAffi'])[0],3)

xy = np.vstack([CIP_prop_df_sasa['HA'], CIP_prop_df_sasa['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = CIP_prop_df_sasa['HA'][idx], CIP_prop_df_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('heavy atom number', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
# ax.set_xlim(0,1000)
ax.set_ylim(0,16)
ax.set_title(f'CASF_subset (N={len(CIP_prop_df_sasa)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('property/CIP_HA_pAffi_density.png', dpi=300, bbox_inches='tight')
plt.close()

## 7.4 SASA
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
# pearsonr = round(stats.pearsonr(CIP_prop_df_sasa['del_sasa'], CIP_prop_df_sasa['-logAffi'])[0],3)
# spearmanr = round(stats.spearmanr(CIP_prop_df_sasa['del_sasa'], CIP_prop_df_sasa['-logAffi'])[0],3)

# xy = np.vstack([CIP_prop_df_sasa['del_sasa'], CIP_prop_df_sasa['-logAffi']])  #按行叠加
# g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
# z = g(xy)  #计算每个xy样本点的概率密度

# # Sort the points by density, so that the densest points are plotted last
# idx = z.argsort()  #对z值排序并返回索引
# y, y_, z = CIP_prop_df_sasa['del_sasa'][idx], CIP_prop_df_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

# fig, ax = plt.subplots(figsize=(6,6))
# ax.scatter(y, y_, s=2, c=z, zorder=2)
# ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
# ax.set_ylabel('Experimental pAffi', fontsize=15)
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)
# # ax.set_xlim(0,1000)
# ax.set_ylim(0,16)
# ax.set_title(f'CASF_subset (N={len(CIP_prop_df_sasa)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
# plt.savefig('property/CIP_SASA_pAffi_density.png', dpi=300, bbox_inches='tight')
# plt.close()

# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(CIP_prop_df_sasa['del_sasa'], CIP_prop_df_sasa['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(CIP_prop_df_sasa['del_sasa'], CIP_prop_df_sasa['-logAffi'])[0],3)

xy = np.vstack([CIP_prop_df_sasa['del_sasa'], CIP_prop_df_sasa['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = CIP_prop_df_sasa['del_sasa'][idx], CIP_prop_df_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,18)
ax.set_ylim(0,15)
ax.set_title(f'CASF_subset (N={len(CIP_prop_df_sasa)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('property/CIP_SASA_pAffi_density_scaled_18.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. training performance
# sum_df = pd.read_csv('/pubhome/xli02/project/PLIM/analysis/20220829_paper/ML/rm_core_all_simi_1/sum.csv', sep='\t')
selected_df = sum_df[sum_df['dataset'].isin(['PIP', 'PLANet', 'PIPUP']) & sum_df['test_type'].isin(['valid', 'train', 'test', 'CASF_v16_minimized', 'CASF_v16_intersected_Uw_minimized'])].copy()

grouped_median = selected_df.groupby(['test_type', 'dataset', 'model_type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_cmx' if row.model_type == 'complex' else f'{row.dataset}_lig_alone' for row in grouped_median.itertuples()]
grouped = selected_df.groupby('test_type')
test_type_to_test_title={'valid':'validation', 'train':'training', 'test':'self_testing', 'CASF_v16_minimized':'CASF_v16', 'CASF_v16_intersected_Uw_minimized':'CASF_v16_intersected_PLANet'}

order = ['PIP', 'PLANet', 'PIPUP']

## 8.1 Rp
metric = 'pearsonr'

out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

### training set
test_type = 'train'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.07 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
# if metric == 'pearsonr':
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rp', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### validation set
test_type = 'valid'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.07 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
# if metric == 'pearsonr':
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rp', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### self_testing set
test_type = 'test'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.07 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
# if metric == 'pearsonr':
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rp', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

## 8.2 Rs
metric = 'spearmanr'
out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

### 8.2.1 training set
test_type = 'train'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.1 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rs', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### 8.2.2 validation set
test_type = 'valid'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.07 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rs', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### 8.2.3 self_testing set
test_type = 'test'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.08 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rs', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

## 8.3 RMSE
metric = 'rmse'
out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

### 8.3.1 training set
test_type = 'train'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.3 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4, 2)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('RMSE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### 8.3.2 validation set
test_type = 'valid'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.1 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4, 2)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('RMSE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### 8.3.3 self_testing set
test_type = 'test'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.2 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.4, 2)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('RMSE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

## 8.4 MAE
metric = 'mae'
out_dir = f'scaled/{metric}'
if not Path(out_dir).exists():
    Path(out_dir).mkdir()

### 8.4.1 training set
test_type = 'train'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.25 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.3, 1.5)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('MAE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### 8.4.2 validation set
test_type = 'valid'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.15 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.3, 1.5)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('MAE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

### 8.4.3 self_testing set
test_type = 'test'
test_df = grouped.get_group(test_type)
fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = order, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on {test_type_to_test_title[test_type]} set', fontsize=15)
vertical_offset = test_df[metric].median() * 0.15 # offset from median for display
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
for i, modl in enumerate(order):
    for tp in ['cmx', 'lig_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
    # ax.set_ylim(min(test_df[metric])-0.2, max(test_df[metric])+0.1)
    ax.set_ylim(0.3, 1.5)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('MAE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'{out_dir}/{metric}_on_{test_type}_set.png', dpi=300, bbox_inches='tight')
plt.close()

