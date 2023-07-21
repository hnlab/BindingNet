import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from collections import defaultdict

if not Path('test_on_PDBbind_hold_out_2019').exists():
    Path('test_on_PDBbind_hold_out_2019').mkdir()

models = ['PDBbind_minimized_v18_subset','PLANet_v18', 'PDBbind_minimized_v18_subset_union_PLANet_v18']
tps = ['cmx', 'lig_alone']

test_dir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/hold_out_2019/test_result'
PDBbind_whole_property = pd.read_csv('../../../3-property_calculate/2-other_property/PDBbind_whole.csv', sep='\t')
PDBbind_hold = pd.read_csv('index_rm_all_simi_1/PDBbind_hold_out_2019_subset.csv',  sep='\t')
# PDBbind_hold_prop = pd.merge(PDBbind_whole_property, PDBbind_hold, on=['pdb_id', '-logAffi'])
PDBbind_hold_prop = PDBbind_whole_property[PDBbind_whole_property['pdb_id'].isin(PDBbind_hold['pdb_id'])].copy()

mean_pred_df = pd.DataFrame.from_dict({})
label_to_res_dict = defaultdict(list)
for model in models:
    for tp in tps:
        res_df = pd.read_csv(f'{test_dir}/test_on_PDBbind_hold_out_2019/{model}_{tp}/test.csv', sep='\t')
        # print(len(res_df_heavy_atom))
        # res_df_with_prop_Mw_pAffi = res_df_with_prop[(res_df_with_prop['y_true'] < 10) & (res_df_with_prop['y_true'] >0) & (res_df_with_prop['mw'] < 800)].copy()
        for i in range(5):
            r2 = r2_score(y_true=res_df['y_true'], y_pred=res_df[f'{model}_{tp}_{i+1}_pred'])
            mae = mean_absolute_error(y_true=res_df['y_true'], y_pred=res_df[f'{model}_{tp}_{i+1}_pred'])
            mse = mean_squared_error(y_true=res_df['y_true'], y_pred=res_df[f'{model}_{tp}_{i+1}_pred'])
            pearsonr = stats.pearsonr(res_df['y_true'], res_df[f'{model}_{tp}_{i+1}_pred'])[0]
            spearmanr = stats.spearmanr(res_df['y_true'], res_df[f'{model}_{tp}_{i+1}_pred'])[0]
            label_to_res_dict[f'{model}_{tp}_{i+1}']=[r2, mae, mse, pearsonr, spearmanr]

        res_df[f'{model}_{tp}_mean'] = res_df[[f'{model}_{tp}_{i+1}_pred' for i in range(5)]].mean(axis=1)
        if mean_pred_df.empty:
            mean_pred_df = res_df[['unique_identify', 'y_true', f'{model}_{tp}_mean']].copy()
        else:
            mean_pred_df = pd.merge(mean_pred_df, res_df[['unique_identify', 'y_true', f'{model}_{tp}_mean']], on=['unique_identify', 'y_true'])

mean_pred_df.to_csv('test_on_PDBbind_hold_out_2019/6models.csv', sep='\t', index=False)

sum_df = pd.DataFrame.from_dict(label_to_res_dict, orient='index', columns=['r2', 'mae', 'mse', 'pearsonr', 'spearmanr']).reset_index()
sum_df.rename(columns={"index": "model_names_test_type"}, inplace=True)
sum_df['rmse'] = np.sqrt(sum_df['mse'])
sum_df['model_tp'] = sum_df['model_names_test_type'].str.rsplit('_', n=1).str[0]
sum_df['type'] = ['cmx' if 'cmx' in m else 'ligand_alone' for m in sum_df['model_tp']]
sum_df['dataset'] = ['_'.join(m.split('_')[:-1]) if 'cmx' in m else '_'.join(m.split('_')[:-2]) for m in sum_df['model_tp']]
sum_df.to_csv('test_on_PDBbind_hold_out_2019/6_models_5_repeats_metrics_heavy_atom_filtered.csv', sep='\t', index=False)


res_dict={}
for model in models:
    for tp in tps:
        r2 = r2_score(y_true=mean_pred_df['y_true'], y_pred=mean_pred_df[f'{model}_{tp}_mean'])
        mae = mean_absolute_error(y_true=mean_pred_df['y_true'], y_pred=mean_pred_df[f'{model}_{tp}_mean'])
        mse = mean_squared_error(y_true=mean_pred_df['y_true'], y_pred=mean_pred_df[f'{model}_{tp}_mean'])
        pearsonr = stats.pearsonr(mean_pred_df['y_true'], mean_pred_df[f'{model}_{tp}_mean'])[0]
        spearmanr = stats.spearmanr(mean_pred_df['y_true'], mean_pred_df[f'{model}_{tp}_mean'])[0]
        res_dict[f'{model}_{tp}']=[r2, mae, mse, pearsonr, spearmanr]

com_sum_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['r2', 'mae', 'mse', 'pearsonr', 'spearmanr']).reset_index()
com_sum_df.rename(columns={"index": "model_names_test_type"}, inplace=True)
com_sum_df['type'] = ['cmx' if 'cmx' in m else 'lig_alone' for m in com_sum_df['model_names_test_type']]
com_sum_df['dataset'] = ['_'.join(m.split('_')[:-1]) if 'cmx' in m else '_'.join(m.split('_')[:-2]) for m in com_sum_df['model_names_test_type']]
com_sum_df.to_csv('test_on_PDBbind_hold_out_2019/6models_metric.csv', sep='\t', index=False)
