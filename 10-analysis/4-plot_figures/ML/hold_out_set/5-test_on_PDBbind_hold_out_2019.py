import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.stats import gaussian_kde

if not Path('test_on_PDBbind_hold_out_2019/scaled').exists():
    Path('test_on_PDBbind_hold_out_2019/scaled').mkdir()

if not Path('test_on_PDBbind_hold_out_2019/property_distribution').exists():
    Path('test_on_PDBbind_hold_out_2019/property_distribution').mkdir()

if not Path('test_on_PDBbind_hold_out_2019/mean_of_5_models/SASA').exists():
    Path('test_on_PDBbind_hold_out_2019/mean_of_5_models/SASA').mkdir(parents=True)


models = ['PDBbind_minimized_v18_subset','PLANet_v18', 'PDBbind_minimized_v18_subset_union_PLANet_v18']
tps = ['cmx', 'lig_alone']

test_dir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019/test_result'

PDBbind_whole_property = pd.read_csv('../../../3-property_calculate/2-other_property/PDBbind_whole.csv', sep='\t')

PDBbind_hold = pd.read_csv('index_rm_all_simi_1/PDBbind_hold_out_2019_subset.csv',  sep='\t')
# PDBbind_hold_prop = pd.merge(PDBbind_whole_property, PDBbind_hold, on=['pdb_id', '-logAffi'])
PDBbind_hold_prop = PDBbind_whole_property[PDBbind_whole_property['pdb_id'].isin(PDBbind_hold['pdb_id'])].copy()

# 1. metrics
sum_df = pd.read_csv('test_on_PDBbind_hold_out_2019/6_models_5_repeats_metrics_heavy_atom_filtered.csv', sep='\t')
grouped_median = sum_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

## 1.1 Rp
metric='pearsonr'
# fig, ax= plt.subplots(figsize=(10,4))
# sns.boxplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, linewidth=2.5)
# sns.swarmplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
# ax.set_title(f'Rp on PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})', fontsize=15)
# # fig.autofmt_xdate()
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2], loc='lower right')
# vertical_offset = sum_df[metric].median() * 0.12 # offset from median for display
# for i, modl in enumerate(models):
#     for tp in ['cmx', 'ligand_alone']:
#         median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
#         if tp == 'cmx':
#             ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
#         else:
#             ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
# ax.set_ylim(0.2,0.6)
# ax.set_xticklabels(['PDBbind_v18_subset', 'BindingNet_v18', 'PDBbind_v18_subset '+r'$\cup{}$' + ' BindingNet_v18'])

# plt.setp(ax.get_legend().get_texts(), fontsize='12')
# plt.setp(ax.get_legend().get_title(), fontsize='12')
# ax.set_xlabel('Dataset', fontsize=15) #
# ax.set_ylabel('Rp', fontsize=15) #
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)

# plt.savefig(f'PDBbind_hold_out_2019/scaled/{metric}_all_5_repeat_models.png', dpi=300, bbox_inches='tight')
# plt.close()

fig, ax= plt.subplots(figsize=(10,4))
sns.boxplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})', fontsize=15)
# fig.autofmt_xdate()

pairs = [(("PLANet_v18", "ligand_alone"), ("PDBbind_minimized_v18_subset", "ligand_alone")), 
    (("PLANet_v18", "cmx"), ("PDBbind_minimized_v18_subset", "cmx")), 
    # (("PIP", "complex"), ("PIPUP", "complex")),
    # (("PIP", "ligand_alone"), ("PIPUP", "ligand_alone")),
    # (("PDBbind", "cmx"), ("DOCK", "cmx")),
    ]
# add_median_labels(ax.axes)
annotator = Annotator(ax, pairs, x="dataset", y=metric, data=sum_df, hue="type", order = models)
annotator.configure(test='t-test_ind').apply_and_annotate()


handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = sum_df[metric].median() * 0.12 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.2,0.6)
ax.set_xticklabels(['PDBbind_v18_subset', 'BindingNet_v18', 'PDBbind_v18_subset '+r'$\cup{}$' + ' BindingNet_v18'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rp', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'test_on_PDBbind_hold_out_2019/scaled/{metric}_all_5_repeat_models_with_sig.png', dpi=300, bbox_inches='tight')
plt.close()

## 1.2 Rs
metric='spearmanr'
fig, ax= plt.subplots(figsize=(10,4))
sns.boxplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = sum_df[metric].median() * 0.15 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.15,0.6)
ax.set_xticklabels(['PDBbind_v18_subset', 'BindingNet_v18', 'PDBbind_v18_subset '+r'$\cup{}$' + ' BindingNet_v18'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rs', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'test_on_PDBbind_hold_out_2019/scaled/{metric}_all_5_repeat_models.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.3 RMSE
metric='rmse'
fig, ax= plt.subplots(figsize=(10,4))
sns.boxplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = sum_df[metric].median() * 0.05 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(1.25,1.6)
ax.set_xticklabels(['PDBbind_v18_subset', 'BindingNet_v18', 'PDBbind_v18_subset '+r'$\cup{}$' + ' BindingNet_v18'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('RMSE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'test_on_PDBbind_hold_out_2019/scaled/{metric}_all_5_repeat_models.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.4 MAE
metric='mae'

# fig, ax= plt.subplots(figsize=(10,4))
# sns.boxplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, linewidth=2.5)
# sns.swarmplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
# ax.set_title(f'MAE on PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})', fontsize=15)
# # fig.autofmt_xdate()
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2], loc='lower right')
# vertical_offset = sum_df[metric].median() * 0.04 # offset from median for display
# for i, modl in enumerate(models):
#     for tp in ['cmx', 'ligand_alone']:
#         median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
#         if tp == 'cmx':
#             ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
#         else:
#             ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
# ax.set_ylim(0.95,1.3)
# ax.set_xticklabels(['PDBbind_v18_subset', 'BindingNet_v18', 'PDBbind_v18_subset '+r'$\cup{}$' + ' BindingNet_v18'])

# plt.setp(ax.get_legend().get_texts(), fontsize='12')
# plt.setp(ax.get_legend().get_title(), fontsize='12')
# ax.set_xlabel('Dataset', fontsize=15) #
# ax.set_ylabel('MAE', fontsize=15) #
# ax.tick_params(axis='x', labelsize= 12)
# ax.tick_params(axis='y', labelsize= 12)

# plt.savefig(f'PDBbind_hold_out_2019/scaled/{metric}_all_5_repeat_models.png', dpi=300, bbox_inches='tight')
# plt.close()

fig, ax= plt.subplots(figsize=(10,4))
sns.boxplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=sum_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})', fontsize=15)
# fig.autofmt_xdate()

pairs = [(("PLANet_v18", "ligand_alone"), ("PDBbind_minimized_v18_subset", "ligand_alone")), 
    (("PLANet_v18", "cmx"), ("PDBbind_minimized_v18_subset", "cmx")), 
    ]
annotator = Annotator(ax, pairs, x="dataset", y=metric, data=sum_df, hue="type", order = models)
annotator.configure(test='t-test_ind').apply_and_annotate()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = sum_df[metric].median() * 0.04 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.95,1.3)
ax.set_xticklabels(['PDBbind_v18_subset', 'BindingNet_v18', 'PDBbind_v18_subset '+r'$\cup{}$' + ' BindingNet_v18'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('MAE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'test_on_PDBbind_hold_out_2019/scaled/{metric}_all_5_repeat_models_with_sig.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. similarity
max_simi_distribution = pd.read_csv('similarity/all_6_models_max_simi_distribution.csv', sep='\t')
# len(max_simi_distribution) # 481

fig, ax= plt.subplots(figsize=(8,6))
sns.kdeplot(max_simi_distribution['PDBbind_v18_subset'])
sns.kdeplot(max_simi_distribution['PLANet_v18'])
sns.kdeplot(max_simi_distribution['PDBbind_v18_subset_union_PLANet_v18'])
plt.xlabel("Tanimoto similarity")
plt.xlim(0,1)
ax.set_title(f'Best similarity distribution among PDBbind_hold_out_2019 and dataset', fontsize=15) #
plt.legend(labels=['PDBbind_v18_subset', 'BindingNet_v18', 'PDBbind_v18_subset' +r'$\cup$' 'BindingNet_v18'], title = "Dataset")

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Tanimoto Similarity', fontsize=15) #
ax.set_ylabel('Density', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'similarity/3_models.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. property distribution of PDBbind_hold_out_2019
PDBbind_hold_prop.reset_index(inplace=True, drop=True)

## 3.1 Mw
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(PDBbind_hold_prop['mw'], PDBbind_hold_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PDBbind_hold_prop['mw'], PDBbind_hold_prop['-logAffi'])[0],3)

xy = np.vstack([PDBbind_hold_prop['mw'], PDBbind_hold_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PDBbind_hold_prop['mw'][idx], PDBbind_hold_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('Molecular Weight (Da)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.set_xlim(0,1000)
ax.set_ylim(0,12)
ax.set_title(f'PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})\npearsonr={pearsonr},spearmanr={spearmanr}', fontsize=15)

ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig('test_on_PDBbind_hold_out_2019/property_distribution/Mw_pAffi_scaled_1000.png', dpi=300, bbox_inches='tight')
plt.close()


# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(PDBbind_hold_prop['mw'], PDBbind_hold_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PDBbind_hold_prop['mw'], PDBbind_hold_prop['-logAffi'])[0],3)

xy = np.vstack([PDBbind_hold_prop['mw'], PDBbind_hold_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PDBbind_hold_prop['mw'][idx], PDBbind_hold_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('Molecular Weight (Da)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.set_xlim(0,1000)
ax.set_ylim(0,16)
ax.set_title(f'PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})\npearsonr={pearsonr},spearmanr={spearmanr}', fontsize=15)

ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig('test_on_PDBbind_hold_out_2019/property_distribution/Mw_pAffi_scaled_1000_16.png', dpi=300, bbox_inches='tight')
plt.close()

## 3.2 HA
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(PDBbind_hold_prop['HA'], PDBbind_hold_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PDBbind_hold_prop['HA'], PDBbind_hold_prop['-logAffi'])[0],3)

xy = np.vstack([PDBbind_hold_prop['HA'], PDBbind_hold_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PDBbind_hold_prop['HA'][idx], PDBbind_hold_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('Heavy atom number', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.set_xlim(0,60)
ax.set_ylim(0,12)
ax.set_title(f'PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})\npearsonr={pearsonr},spearmanr={spearmanr}', fontsize=15)

ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig('test_on_PDBbind_hold_out_2019/property_distribution/HA_pAffi_scaled_1000.png', dpi=300, bbox_inches='tight')
plt.close()

## 3.3 SASA
PDBbind_whole_sasa = pd.read_csv('../../../3-property_calculate/1-sasa/PDBbind_whole_sasa.csv')
# len(PDBbind_whole_sasa) # 17176

PLANet_whole_sasa = pd.read_csv('../../../3-property_calculate/1-sasa/PLANet_property_sasa.csv')
# len(PLANet_whole_sasa) # 69813

sasa_info = pd.concat([PDBbind_whole_sasa, PLANet_whole_sasa])
# len(sasa_info) # 86989

### 3.3.1 PDBbind_hold_out
PDBbind_hold_prop_sasa = pd.merge(PDBbind_hold_prop, PDBbind_whole_sasa.rename(columns={'unique_identity':'pdb_id'}), on=['pdb_id']) # 485
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(PDBbind_hold_prop_sasa['del_sasa'], PDBbind_hold_prop_sasa['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PDBbind_hold_prop_sasa['del_sasa'], PDBbind_hold_prop_sasa['-logAffi'])[0],3)

xy = np.vstack([PDBbind_hold_prop_sasa['del_sasa'], PDBbind_hold_prop_sasa['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PDBbind_hold_prop_sasa['del_sasa'][idx], PDBbind_hold_prop_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.set_xlim(0, 18)
ax.set_ylim(0,15)
ax.set_title(f'PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop_sasa)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('test_on_PDBbind_hold_out_2019/property_distribution/SASA_pAffi_scaled_18.png', dpi=300, bbox_inches='tight')
plt.close()

### 3.3.2 PDBbind_v18_subset
PDBbind_v18_subset = pd.read_csv('index_rm_all_simi_1/PDBbind_v18_subset_rm_simi_1.csv', sep='\t')
# len(PDBbind_v18_subset) # 5371
PDBbind_v18_subset_sasa = pd.merge(PDBbind_v18_subset, sasa_info.rename(columns={'unique_identity':'pdb_id'}), on=['pdb_id'])
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(PDBbind_v18_subset_sasa['del_sasa'], PDBbind_v18_subset_sasa['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PDBbind_v18_subset_sasa['del_sasa'], PDBbind_v18_subset_sasa['-logAffi'])[0],3)

xy = np.vstack([PDBbind_v18_subset_sasa['del_sasa'], PDBbind_v18_subset_sasa['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PDBbind_v18_subset_sasa['del_sasa'][idx], PDBbind_v18_subset_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.set_xlim(0, 18)
ax.set_ylim(0,15)
ax.set_title(f'PDBbind_v18_subset (N={len(PDBbind_v18_subset_sasa)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('test_on_PDBbind_hold_out_2019/property_distribution/SASA_pAffi_scaled_18_PDBbind_v18_subset.png', dpi=300, bbox_inches='tight')
plt.close()

### 3.3.3 PLANet_v18
PLANet_v18 = pd.read_csv('index_rm_all_simi_1/PLANet_v18_rm_simi_1.csv', sep='\t')
# len(PLANet_v18) # 63604
PLANet_v18_sasa = pd.merge(PLANet_v18, sasa_info.rename(columns={'unique_identity':'unique_identify'}), on=['unique_identify'])
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)

pearsonr = round(stats.pearsonr(PLANet_v18_sasa['del_sasa'], PLANet_v18_sasa['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PLANet_v18_sasa['del_sasa'], PLANet_v18_sasa['-logAffi'])[0],3)

xy = np.vstack([PLANet_v18_sasa['del_sasa'], PLANet_v18_sasa['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PLANet_v18_sasa['del_sasa'][idx], PLANet_v18_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.set_xlim(0, 18)
ax.set_ylim(0,15)
ax.set_title(f'BindingNet_v18 (N=63604)\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('test_on_PDBbind_hold_out_2019/property_distribution/SASA_pAffi_scaled_18_BindingNet_v18.png', dpi=300, bbox_inches='tight')
plt.close()


### 3.3.4 PIPUP
PIPUP_v18 = pd.read_csv('index_rm_all_simi_1/PDBbind_v18_subset_union_PLANet_rm_simi_1.csv', sep='\t')
# len(PIPUP_v18) # 68975
PIPUP_v18_sasa = pd.merge(PIPUP_v18, sasa_info.rename(columns={'unique_identity':'unique_identify'}), on=['unique_identify'])
# RIP_minimized_PCV.reset_index(drop=True, inplace=True)
pearsonr = round(stats.pearsonr(PIPUP_v18_sasa['del_sasa'], PIPUP_v18_sasa['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PIPUP_v18_sasa['del_sasa'], PIPUP_v18_sasa['-logAffi'])[0],3)

xy = np.vstack([PIPUP_v18_sasa['del_sasa'], PIPUP_v18_sasa['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PIPUP_v18_sasa['del_sasa'][idx], PIPUP_v18_sasa['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.set_xlim(0, 18)
ax.set_ylim(0,15)
ax.set_title('PDBbind_v18_subset '+r'$\cup{}$' + f' BindingNet_v18 (N=68975)\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('test_on_PDBbind_hold_out_2019/property_distribution/SASA_pAffi_scaled_18_PIPUP_v18.png', dpi=300, bbox_inches='tight')
plt.close()

### 3.3.5 training set mean_predicted-buried SASA
mean_pred_df = pd.read_csv('test_on_PDBbind_hold_out_2019/6models.csv', sep='\t')
com_sum_df=pd.read_csv('test_on_PDBbind_hold_out_2019/6models_metric.csv', sep='\t')
simple_dict = {'PDBbind_minimized_v18_subset':'PDBbind_v18_subset', 'PDBbind_minimized_v18_subset_union_PLANet_v18':'PDBbind_v18_subset '+r'$\cup{}$' + ' BindingNet_v18','PLANet_v18':'BindingNet_v18'}
mean_pred_df_with_prop = pd.merge(mean_pred_df.round({'y_true': 2}), PDBbind_hold_prop.rename(columns={'pdb_id':'unique_identify', '-logAffi':'y_true'}), on=['unique_identify', 'y_true'])
mean_pred_df_with_prop_sasa = pd.merge(mean_pred_df_with_prop, PDBbind_whole_sasa.rename(columns={'unique_identity':'unique_identify'}), on=['unique_identify'])

fig = plt.figure(figsize=(16,16))
for i, model in enumerate(models):
    for j, tp in enumerate(tps):
        # print((i+1)+6*j)
        ax = fig.add_subplot(3, 3, (i+1)+3*j)
        # fig, ax = plt.subplots(figsize=(6,6))

        y_true = mean_pred_df_with_prop_sasa['del_sasa']
        y_pred = mean_pred_df_with_prop_sasa[f'{model}_{tp}_mean']

        xy = np.vstack([y_true.T, y_pred.T])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        y_true_, y_pred_, z = y_true[idx], y_pred[idx], z[idx]

        #https://github.com/hnlab/handbook/blob/41ad374cd0f9dc3ef882a7724eaac3d1f748fc05/0-General-computing-skills/MISC/vsfig.py#L83-L134
        # fig, ax = plt.subplots()
        ax.scatter(y_true_, y_pred_, s=2, c=z, zorder=2)

        # sns.scatterplot(x='mw', y=f'{model}_{tp}_mean', data=mean_pred_df_with_prop)

        # if model=='PDBbind_minimized_intersected_Uw':
        #     ax.axvline(x=800, color='black', linestyle='--')

        ax.set_xlim(0,18)
        ax.set_ylim(0,15)
        ax.tick_params(axis='x', labelsize= 12)
        ax.tick_params(axis='y', labelsize= 12)

        ax.set_ylabel(f'mean_predicted pAffi', fontsize=15)
        ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
        ax.set_title(f'{simple_dict[model]}_{tp}\nRp={round(stats.pearsonr(mean_pred_df_with_prop_sasa[f"{model}_{tp}_mean"], mean_pred_df_with_prop_sasa["del_sasa"])[0], 3)}, Rs={round(stats.spearmanr(mean_pred_df_with_prop_sasa[f"{model}_{tp}_mean"], mean_pred_df_with_prop_sasa["del_sasa"])[0], 3)}', fontsize=15)
plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.93, 
                    wspace=0.2, 
                    hspace=0.3)
plt.suptitle(f'Relationship between mean_predict and buried SASA on PDBbind_hold_out_2019 set (N={len(PDBbind_hold_prop)})', fontsize=15)
plt.savefig('test_on_PDBbind_hold_out_2019/mean_of_5_models/SASA/scatter_for_12_models.png', dpi=300, bbox_inches='tight')
plt.close()


#### seperate: PDBbind_v18_subset_cmx
model = 'PDBbind_minimized_v18_subset'
tp='cmx'

fig, ax = plt.subplots(figsize=(6,6))

y_true = mean_pred_df_with_prop_sasa['del_sasa']
y_pred = mean_pred_df_with_prop_sasa[f'{model}_{tp}_mean']

xy = np.vstack([y_true.T, y_pred.T])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
y_true_, y_pred_, z = y_true[idx], y_pred[idx], z[idx]

#https://github.com/hnlab/handbook/blob/41ad374cd0f9dc3ef882a7724eaac3d1f748fc05/0-General-computing-skills/MISC/vsfig.py#L83-L134
# fig, ax = plt.subplots()
ax.scatter(y_true_, y_pred_, s=2, c=z, zorder=2)

# sns.scatterplot(x='mw', y=f'{model}_{tp}_mean', data=mean_pred_df_with_prop)

# if model=='PDBbind_minimized_intersected_Uw':
#     ax.axvline(x=800, color='black', linestyle='--')

ax.set_xlim(0,18)
ax.set_ylim(0,15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

ax.set_ylabel(f'mean_predicted pAffi', fontsize=15)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_title(f'{simple_dict[model]}_{tp} on PDBbind_hold_out_2019 set\nRp={round(stats.pearsonr(mean_pred_df_with_prop_sasa[f"{model}_{tp}_mean"], mean_pred_df_with_prop_sasa["del_sasa"])[0], 3)}, Rs={round(stats.spearmanr(mean_pred_df_with_prop_sasa[f"{model}_{tp}_mean"], mean_pred_df_with_prop_sasa["del_sasa"])[0], 3)}', fontsize=15)
plt.savefig('test_on_PDBbind_hold_out_2019/mean_of_5_models/SASA/SASA_pAffi_PDBbind_v18_subset_cmx_scaled_18.png', dpi=300, bbox_inches='tight')
plt.close()

#### seperate: PLANet_v18_cmx
model = 'PLANet_v18'
tp='cmx'

fig, ax = plt.subplots(figsize=(6,6))

y_true = mean_pred_df_with_prop_sasa['del_sasa']
y_pred = mean_pred_df_with_prop_sasa[f'{model}_{tp}_mean']

xy = np.vstack([y_true.T, y_pred.T])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
y_true_, y_pred_, z = y_true[idx], y_pred[idx], z[idx]

#https://github.com/hnlab/handbook/blob/41ad374cd0f9dc3ef882a7724eaac3d1f748fc05/0-General-computing-skills/MISC/vsfig.py#L83-L134
# fig, ax = plt.subplots()
ax.scatter(y_true_, y_pred_, s=2, c=z, zorder=2)

# sns.scatterplot(x='mw', y=f'{model}_{tp}_mean', data=mean_pred_df_with_prop)

# if model=='PDBbind_minimized_intersected_Uw':
#     ax.axvline(x=800, color='black', linestyle='--')

ax.set_xlim(0,18)
ax.set_ylim(0,15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

ax.set_ylabel(f'mean_predicted pAffi', fontsize=15)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_title(f'{simple_dict[model]}_{tp} on PDBbind_hold_out_2019 set\nRp={round(stats.pearsonr(mean_pred_df_with_prop_sasa[f"{model}_{tp}_mean"], mean_pred_df_with_prop_sasa["del_sasa"])[0], 3)}, Rs={round(stats.spearmanr(mean_pred_df_with_prop_sasa[f"{model}_{tp}_mean"], mean_pred_df_with_prop_sasa["del_sasa"])[0], 3)}', fontsize=15)
plt.savefig('test_on_PDBbind_hold_out_2019/mean_of_5_models/SASA/SASA_pAffi_PLANet_v18_cmx_scaled_18.png', dpi=300, bbox_inches='tight')
plt.close()

