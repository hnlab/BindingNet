import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

wrkdir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim'
label_to_res_dict = defaultdict(list)

log_dir_names = ['PDBbind_minimized','PDBbind_minimized_intersected_Uw','PLANet_Uw', 'PDBbind_minimized_intersected_Uw_union_Uw', 'PDBbind_minimized_union_Uw', 'PLANet_all'] # plot order
mdl_types = ['complex_6A', 'lig_alone']
test_types = ['valid', 'train', 'test'] #change by the order in log

for log_dir_name in log_dir_names:
    for mdl_type in mdl_types:
        if log_dir_name in ['PDBbind_minimized', 'PDBbind_minimized_intersected_Uw']:
            log_dir =  f'{wrkdir}/PDBbind/pdbbind_v2019/minimized/2-train/true_lig_alone/test/whole_set/{log_dir_name}/{mdl_type}/log'
        else:
            log_dir = f'{wrkdir}/3-test/scripts/true_lig_alone_modify_dists/{mdl_type}/whole_set/{log_dir_name}/log/'

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
                        pearsonr = float(line.split(',')[3].split('(')[1])
                        spearmanr =float(line.split(',')[5].split('=')[1])
                        label_to_res_dict[f'{log_dir_name}_{mdl_type}_{i+1}_{line.split()[2]}']=[R2, mae, mse, pearsonr, spearmanr]

sum_df = pd.DataFrame.from_dict(label_to_res_dict, orient='index', columns=['R2', 'mae', 'mse', 'pearsonr', 'spearmanr']).reset_index()
sum_df.rename(columns={"index": "model_names_test_type"}, inplace=True)
sum_df['model_names'] = sum_df['model_names_test_type'].str.rsplit('_', n=1).str[0]
sum_df['test_type'] = sum_df['model_names_test_type'].str.rsplit('_', n=1).str[1]
sum_df['dataset'] = sum_df['model_names'].str.rsplit('_', n=3).str[0]
sum_df['model_type'] = ['complex' if 'complex' in m else 'ligand_alone' for m in sum_df['model_names']]
sum_df.to_csv(f'{wrkdir}/5-evaluation/PDBbind_vs_PLANet_whole_set/PDBbind_vs_PLANet_whole_set.csv', sep='\t', index=False)

grouped_median = sum_df.groupby(['test_type', 'dataset', 'model_type']).median().reset_index()
grouped_median['model_name'] = [f'{row.dataset}_cmx' if row.model_type == 'complex' else f'{row.dataset}_lig_alone' for row in grouped_median.itertuples()]
grouped_median.to_csv(f'{wrkdir}/5-evaluation/PDBbind_vs_PLANet_whole_set/PDBbind_vs_PLANet_whole_set_median.csv', sep='\t', index=False)

grouped = sum_df.groupby('test_type')
for metric in ['R2', 'mae', 'mse', 'pearsonr', 'spearmanr']:
    out_dir = f'{wrkdir}/5-evaluation/PDBbind_vs_PLANet_whole_set/{metric}'
    if not Path(out_dir).exists():
        Path(out_dir).mkdir()
    for test_type in test_types:
        test_df = grouped.get_group(test_type)
        fig, ax= plt.subplots()
        sns.boxplot(x="dataset", y=metric, data=test_df, hue="model_type", order = log_dir_names, linewidth=2.5)
        sns.swarmplot(x="dataset", y=metric, data=test_df, hue="model_type", order = log_dir_names, size = 4, dodge=True, edgecolor="black", linewidth=0.7)
        ax.set_title(f'{metric} on {test_type} set')
        fig.autofmt_xdate()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2])
        vertical_offset = test_df[metric].median() * 0.03 # offset from median for display
        for i, modl in enumerate(log_dir_names):
            for tp in ['cmx', 'lig_alone']:
                median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{modl}_{tp}') & (grouped_median['test_type'] == test_type)][metric].values[0], 3)
                if tp == 'cmx':
                    ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
                else:
                    ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
        if metric in ['R2', 'pearsonr', 'spearmanr']:
            ax.set_ylim(min(test_df[metric])-0.05, min(max(test_df[metric])+0.05, 1))
        plt.savefig(f'{out_dir}/{metric}_on_{test_type}.png', dpi=300, bbox_inches='tight')
        plt.close()

#performance_inner_dataset
data_grouped = sum_df.groupby('dataset')
for metric in ['R2', 'mae', 'mse', 'pearsonr', 'spearmanr']:
    out_dir = f'{wrkdir}/5-evaluation/PDBbind_vs_PLANet_whole_set/{metric}/performance_inner_dataset'
    if not Path(out_dir).exists():
        Path(out_dir).mkdir()
    for test_data in log_dir_names:
        data_df = data_grouped.get_group(test_data)
        fig, ax= plt.subplots()
        sns.boxplot(x="test_type", y=metric, data=data_df, hue="model_type", order = ['train', 'valid', 'test'], linewidth=2.5)
        sns.swarmplot(x="test_type", y=metric, data=data_df, hue="model_type", order = ['train', 'valid', 'test'], size = 4, dodge=True, edgecolor="black", linewidth=0.7)
        ax.set_title(f'{metric} on {test_data} set')
        fig.autofmt_xdate()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2])
        vertical_offset = data_df[metric].median() * 0.07 # offset from median for display
        for i, test_tp in enumerate(['train', 'valid', 'test']):
            for tp in ['cmx', 'lig_alone']:
                median_metric = round(grouped_median.loc[(grouped_median['model_name'] == f'{test_data}_{tp}') & (grouped_median['test_type'] == test_tp)][metric].values[0], 3)
                if tp == 'cmx':
                    ax.text(i-.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small', weight='semibold')
                else:
                    ax.text(i+.2, median_metric-vertical_offset, median_metric, horizontalalignment='center',size='small',weight='semibold')
        if metric in ['R2', 'pearsonr', 'spearmanr']:
            ax.set_ylim(min(data_df[metric])-0.05, min(max(data_df[metric])+0.05, 1))
        plt.savefig(f'{out_dir}/{metric}_on_{test_data}.png', dpi=300, bbox_inches='tight')
        plt.close()
