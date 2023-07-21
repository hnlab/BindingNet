import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

if not Path('scaled').exists():
    Path('scaled').mkdir()

models = ['PDBbind_minimized_v18_subset','PLANet_v18', 'PDBbind_minimized_v18_subset_union_PLANet_v18']
tps=['cmx', 'lig_alone']
tps_dict = {'cmx': 'complex_6A', 'lig_alone': 'lig_alone'}

# mean_pred_df = pd.DataFrame.from_dict({})
label_to_res_dict = defaultdict(list)
for model in models:
    for tp in tps:
        for repeat_num in range(5):
            for test_set in ['train', 'valid', 'test']:
                if model in ['PDBbind_minimized_v18_subset']:
                    res_df = pd.read_csv(f'/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/test_result/diff_split/{model}/{tps_dict[tp]}/{repeat_num+1}/{test_set}.csv', sep='\t')
                else:
                    res_df = pd.read_csv(f'/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_results/true_lig_alone_modify_dists/epoch_500_shuffle_true/diff_split/whole_set/{model}/{tps_dict[tp]}/{repeat_num+1}/{test_set}.csv', sep='\t')

                r2 = r2_score(y_true=res_df['y_true'], y_pred=res_df[f'y_pred'])
                mae = mean_absolute_error(y_true=res_df['y_true'], y_pred=res_df[f'y_pred'])
                mse = mean_squared_error(y_true=res_df['y_true'], y_pred=res_df[f'y_pred'])
                rmse = np.sqrt(mse)
                pearsonr = stats.pearsonr(res_df['y_true'], res_df[f'y_pred'])[0]
                spearmanr = stats.spearmanr(res_df['y_true'], res_df[f'y_pred'])[0]
                label_to_res_dict[f'{model}_{tp}_{repeat_num+1}_{test_set}']=[r2, mae, mse, rmse, pearsonr, spearmanr]

training_sum_df = pd.DataFrame.from_dict(label_to_res_dict, orient='index', columns=['r2', 'mae', 'mse', 'rmse','pearsonr', 'spearmanr']).reset_index()
training_sum_df.rename(columns={"index": "model_names_test_type_repeat_set"}, inplace=True)
training_sum_df['model_tp'] = training_sum_df['model_names_test_type_repeat_set'].str.rsplit('_', n=2).str[0]
training_sum_df['test_set'] = training_sum_df['model_names_test_type_repeat_set'].str.rsplit('_', n=2).str[2]
training_sum_df['type'] = ['cmx' if 'cmx' in m else 'ligand_alone' for m in training_sum_df['model_tp']]
training_sum_df['dataset'] = ['_'.join(m.split('_')[:-1]) if 'cmx' in m else '_'.join(m.split('_')[:-2]) for m in training_sum_df['model_tp']]

# 1. Rp
metric = 'pearsonr'

## 1.1 train
test_set = 'train'
test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on training set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = training_sum_df[metric].median() * 0.1 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rp', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.2 valid
test_set = 'valid'
test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on validation set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = training_sum_df[metric].median() * 0.1 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rp', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 1.3 test
test_set = 'test'
test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rp on self_testing set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = training_sum_df[metric].median() * 0.11 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rp', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Rs
metric = 'spearmanr'

# 2.1 train
test_set = 'train'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on training set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = training_sum_df[metric].median() * 0.1 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])


plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rs', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.2 valid
test_set = 'valid'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on validation set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = training_sum_df[metric].median() * 0.1 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rs', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 2.3 test
test_set = 'test'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'Rs on self_testing set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], loc='lower right')
vertical_offset = training_sum_df[metric].median() * 0.11 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('Rs', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. MAE
metric = 'mae'

# 3.1 train
test_set = 'train'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on training set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
vertical_offset = training_sum_df[metric].median() * 0.2 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.3,1.2)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('MAE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.2 valid
test_set = 'valid'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on validation set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
vertical_offset = training_sum_df[metric].median() * 0.2 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.3,1.2)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('MAE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.3 test
test_set = 'test'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'MAE on self_testing set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
vertical_offset = training_sum_df[metric].median() * 0.2 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.3,1.2)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('MAE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. RMSE
metric = 'rmse'

# 4.1 train
test_set = 'train'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on training set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
vertical_offset = training_sum_df[metric].median() * 0.15 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1.5)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('RMSE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.2 valid
test_set = 'valid'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on validation set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
vertical_offset = training_sum_df[metric].median() * 0.15 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1.5)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('RMSE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()

# 4.3 test
test_set = 'test'

test_df = training_sum_df[training_sum_df['test_set'] == test_set].copy()
grouped_median = test_df.groupby(['dataset', 'type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_{row.type}' for row in grouped_median.itertuples()]

fig, ax= plt.subplots(figsize=(7,4))
sns.boxplot(x="dataset", y=metric, data=test_df, hue="type", order = models, linewidth=2.5)
sns.swarmplot(x="dataset", y=metric, data=test_df, hue="type", order = models, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
ax.set_title(f'RMSE on self_testing set', fontsize=15)
# fig.autofmt_xdate()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2])
vertical_offset = training_sum_df[metric].median() * 0.15 # offset from median for display
for i, modl in enumerate(models):
    for tp in ['cmx', 'ligand_alone']:
        median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}')][metric].values[0], 3)
        if tp == 'cmx':
            ax.text(i-.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
        else:
            ax.text(i+.2, median_metric+vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
ax.set_ylim(0.4,1.5)
ax.set_xticklabels(['PDBbind_subset', 'BindingNet', 'PDBbind_subset'+r'$\cup{}$' + 'BindingNet'])

plt.setp(ax.get_legend().get_texts(), fontsize='12')
plt.setp(ax.get_legend().get_title(), fontsize='12')
ax.set_xlabel('Dataset', fontsize=15) #
ax.set_ylabel('RMSE', fontsize=15) #
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)

plt.savefig(f'scaled/{metric}_on_{test_set}.png', dpi=300, bbox_inches='tight')
plt.close()
