'''
input:`bak/6-before_consider_all_reasonable_affi_rather_best_affi/PLANet_all_modified.csv` or `index/PLANet_all_no_affi.csv`
'''

import pandas as pd
import numpy as np
from collections import defaultdict

wrkdir = '/home/xli/Documents/projects/ChEMBL-scaffold'

# 1. 读取之前生成的index file，获取`unique_identify`及其它信息，只需改变对应`unique_identify`活性值即可
# PLANet_all_idx = f'{wrkdir}/v2019_dataset/index/PLANet_all_modified.csv'
PLANet_all_idx = f'{wrkdir}/v2019_dataset/index/bak/6-before_consider_all_reasonable_affi_rather_best_affi/PLANet_all_modified.csv'
PLANet_all_df = pd.read_csv(PLANet_all_idx, sep='\t')
PLANet_all_df['target_compnd'] = [f'{row.Target_chembl_id}_{row.Similar_compnd_name}' for row in PLANet_all_df.itertuples()]

# 2. 重新query ChEMBL数据库，获得`assay_id` & `assay_type`

# 3. 根据index file中的`target_compound`生成对应的活性文件
# 41 min
compnd_to_affi = defaultdict(list)
column_names = ['target_compnd', 'standard_type', 'standard_relation', 'standard_value', 'standard_units', 'assay_chembl_id', 'assay_type']
multiple_affi_target_cpnd = []
filtered_dfs_list = []
i = 0
for target_cpnd in set(PLANet_all_df['target_compnd']):
    target = target_cpnd.split('_')[0]
    compnd = target_cpnd.split('_')[1]
    MolWt = PLANet_all_df[PLANet_all_df['target_compnd'] == target_cpnd]['MolWt'].values[0]
    activity_f = f'{wrkdir}/pipeline/pipeline_2/10.0-Requery_assay_id/query_results/tsv/web_client_{target}-activity.tsv'
    activity_df = pd.read_csv(activity_f, sep='\t')
    affi_compnd_df = activity_df[activity_df['molecule_chembl_id']==compnd]

    ## 3.1 先把units不为`nM`/`ug.mL-1`, 或type不为`B`/`F`的去除
    filter_units_and_assay_type_affi_df = affi_compnd_df.query('standard_units in ["nM", "ug.mL-1"] and assay_type in ["B", "F"]').copy()
    if len(filter_units_and_assay_type_affi_df) == 0:
        print(f"{target_cpnd} has no affinity units " \
            f"with 'nM' or 'ug.mL-1' unit, Or has no assay_type with 'B' or 'F', " \
            f"Let 'Affinity' = 'No data'.")
        compnd_to_affi[target_cpnd] = ['No data']
        continue

    ## 3.2 `B` > `F`: assay_type中若有`B`，则只考虑`B`; 若没有，则用`F`
    if 'B' in set(filter_units_and_assay_type_affi_df['assay_type']):
        assay_type_filtered_df = filter_units_and_assay_type_affi_df.query('assay_type == "B"').copy()
    else:
        print(f"{target_cpnd} only has affinity " \
            f"data with assay_type 'F'.")
        assay_type_filtered_df = filter_units_and_assay_type_affi_df.query('assay_type == "F"').copy()

    ## 3.3 `=` > `<`: relation中若有`=`，则只考虑`=`; 若没有，则用`<`
    if '=' in set(assay_type_filtered_df['standard_relation']):
        relation_filtered_df = assay_type_filtered_df.query('standard_relation == "="').copy()
    else:
        print(f"{target_cpnd} only has affinity " \
            f"data with relation '<'.")
        relation_filtered_df = assay_type_filtered_df.query('standard_relation == "<"').copy()

    for relation_row in relation_filtered_df.itertuples():
        if relation_row.standard_units == 'ug.mL-1':
            relation_filtered_df.loc[relation_row.Index, 'standard_value'] = relation_row.standard_value*1000000/MolWt 
            relation_filtered_df.loc[relation_row.Index, 'standard_units'] = 'nM'
        affi = f"{relation_row.standard_type} " \
        f"{relation_row.standard_relation} " \
        f"{relation_row.standard_value} " \
        f"{relation_row.standard_units}"
        if affi not in compnd_to_affi[target_cpnd]:
            compnd_to_affi[target_cpnd].append(affi)
    
    if len(compnd_to_affi[target_cpnd]) > 1:
        multiple_affi_target_cpnd.append(target_cpnd)
    relation_filtered_df['target_compnd']= [f'{row.target_chembl_id}_{row.molecule_chembl_id}' for row in relation_filtered_df.itertuples()]
    filtered_dfs_list.append(relation_filtered_df[column_names])
    if i % 1000 == 0:
        print(f'complete {i}')
    i = i + 1

target_compnd_affi_all_df = pd.concat(filtered_dfs_list, ignore_index=True)
target_compnd_affi_all_df.to_csv(f'{wrkdir}/pipeline/pipeline_2/10-deal_with_affinity/target_compnd_affi_all_info.csv', sep='\t', index=False)

# target_compnd_affi_all_df['except_assay_id'] = [f'{row.target_compnd}-{row.standard_type}-{row.standard_relation}-{row.standard_value}-{row.standard_units}-{row.assay_type}' for row in target_compnd_affi_all_df.itertuples()]
# len(target_compnd_affi_all_df.groupby('except_assay_id'))

## 3.4 correct affinity manually for `-logAffi` < 0 /> 12
target_compnd_affi_all_df['-logAffi'] = [-np.log10(float(affi) * 10 ** -9) for affi in target_compnd_affi_all_df['standard_value']]
target_compnd_affi_all_df=target_compnd_affi_all_df.round({'-logAffi':2})

target_compnd_affi_all_df[target_compnd_affi_all_df['-logAffi']<0][column_names + ['-logAffi']].to_csv(f'{wrkdir}/pipeline/pipeline_2/10-deal_with_affinity/affi_lower_0.csv', sep='\t', index=False)
#check manually -> `target_compnd_affi_all_info_corrected_manually.csv`

## 3.5 Remove `-logAffi`<0
all_affi_corrected_df = pd.read_csv(f'{wrkdir}/pipeline/pipeline_2/10-deal_with_affinity/target_compnd_affi_all_info_corrected_manually.csv', sep='\t')
all_affi_corrected_df['-logAffi'] = [-np.log10(float(affi) * 10 ** -9) for affi in all_affi_corrected_df['standard_value']]
all_affi_corrected_rm_extrem_min_df = all_affi_corrected_df[all_affi_corrected_df['-logAffi'] > 0].copy()

## 3.6 Remove `relation`==`<` and `value` > 10 nM -> `target_compnd_affi_all_info_Corrected_manually_Rm_extrem_min_Rm_part_less_sign.csv`
# len(all_affi_corrected_rm_extrem_min_df[(all_affi_corrected_rm_extrem_min_df['standard_relation']=='<') & (all_affi_corrected_rm_extrem_min_df['standard_value']>10)]) #1178
all_affi_Corrected_Rm_extrem_min_Rm_part_less_sign_df = all_affi_corrected_rm_extrem_min_df[~((all_affi_corrected_rm_extrem_min_df['standard_relation']=='<') & (all_affi_corrected_rm_extrem_min_df['standard_value']>10))].copy()
all_affi_Corrected_Rm_extrem_min_Rm_part_less_sign_df.to_csv(f'{wrkdir}/pipeline/pipeline_2/10-deal_with_affinity/target_compnd_affi_all_info_Corrected_manually_Rm_extrem_min_Rm_part_less_sign.csv', sep='\t', index=False)

## 3.7 `Kd>Ki>IC50>EC50` for machine learning only -> `target_compnd_affi_only_one_type_for_ML.csv`
#`only_one_type_df`: 为用于机器学习时，按`Kd>Ki>IC50>EC50`的顺序只保留了优先的活性类型; 但是在SAR分析时，应优先考虑是否在同一assay中
# 10 min
only_one_type_list = []
i=0
for target_compnd in set(all_affi_Corrected_Rm_extrem_min_Rm_part_less_sign_df['target_compnd']):
    target_compnd_df = all_affi_Corrected_Rm_extrem_min_Rm_part_less_sign_df[all_affi_Corrected_Rm_extrem_min_Rm_part_less_sign_df['target_compnd']==target_compnd]
    if 'Kd' in set(target_compnd_df['standard_type']):
        filtered_df = target_compnd_df.query('standard_type == "Kd"')
    elif 'Ki' in set(target_compnd_df['standard_type']):
        filtered_df = target_compnd_df.query('standard_type == "Ki"')
    elif 'IC50' in set(target_compnd_df['standard_type']):
        filtered_df =  target_compnd_df.query('standard_type == "IC50"')
    else:
        filtered_df = target_compnd_df.query('standard_type == "EC50"')
    only_one_type_list.append(filtered_df)
    if i % 5000 == 0:
        print(f'complete {i}.')
    i = i+1
only_one_type_df=pd.concat(only_one_type_list, ignore_index=True)
# len(only_one_type_df) #92153
# len(set(only_one_type_df['target_compnd'])) #69826
only_one_type_df.to_csv(f'{wrkdir}/pipeline/pipeline_2/10-deal_with_affinity/target_compnd_affi_only_one_type_for_ML.csv', sep='\t', index=False)

# 4. Combine `PLANet_all_no_affi` and ~~`affi_df_for_ml`~~ `only_one_type_df`(for ML)
## 4.1 去除之前index file中的(best)活性信息 -> `PLANet_all_no_affi.csv`
PLANet_all_Rm_affi_column_df = PLANet_all_df.drop(columns=['Affinity', 'Activity_id']).copy()
PLANet_all_Rm_affi_column_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_all_no_affi.csv', sep='\t', index=False)

## 4.2 merge -> `PLANet_all_Consider_assay_type_Rm_less_than_0_Rm_part_less_sign_All_reasonable_affi_containing_NA.csv`
affi_df_for_ml = only_one_type_df[['target_compnd', '-logAffi', 'standard_type', 'assay_chembl_id', 'assay_type']].copy() #没有保留`standard_relation`，之后可以添加该信息到总的list中，以供参考
PLANet_all_affi_df = pd.merge(PLANet_all_Rm_affi_column_df, affi_df_for_ml, on='target_compnd', how="left")
PLANet_all_affi_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_all_Consider_assay_type_Rm_less_than_0_Rm_part_less_sign_All_reasonable_affi_containing_NA.csv', sep='\t', index=False)

## 4.3 Remove NA for PLANet_all, and obtain `unique_identify` -> `PLANet_all_dealt.csv`
PLANet_all_affi_rm_NA_df = PLANet_all_affi_df[PLANet_all_affi_df['-logAffi'].notna()].copy()
PLANet_all_affi_rm_NA_df['unique_identify'] = [f'{row.Target_chembl_id}_{row.Cry_lig_name}_{row.Similar_compnd_name}' for row in PLANet_all_affi_rm_NA_df.itertuples()]
PLANet_all_affi_rm_NA_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_all_dealt.csv', sep='\t', index=False)

## 4.4 `PLANet_all_final.csv`
PLANet_all_simple_df = PLANet_all_affi_rm_NA_df[['unique_identify', '-logAffi']]
PLANet_all_simple_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_all_final.csv', sep = "\t", index = False)
all_grouped = PLANet_all_simple_df.groupby('unique_identify')
all_median_df = all_grouped['-logAffi'].median().reset_index()
all_median_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_all_final_median.csv', sep='\t', index=False)

## 4.5 PLANet_all_median_affinity_only -> add other information
# 漏掉了assay相关的信息
PLANet_all_Rm_affi_column_df['unique_identify'] = [f'{row.Target_chembl_id}_{row.Cry_lig_name}_{row.Similar_compnd_name}' for row in PLANet_all_Rm_affi_column_df.itertuples()]
all_median_more_info_df = pd.merge(all_median_df, PLANet_all_Rm_affi_column_df, on='unique_identify', how="left")
all_median_more_info_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_all_dealt_median.csv', sep='\t', index=False)

# 5. PLANet_Uw
## choose best pose first: `PLANet_all_Rm_affi_column_df` -> `PLANet_Uw_no_affi`
## `pd.merge(PLANet_Uw_no_affi, affi_df_for_ml, on='target_compnd', how="left")`
## 5.1 `PLANet_all_Rm_affi_column_df` -> `PLANet_Uw_no_affi.csv`
grouped = PLANet_all_Rm_affi_column_df.groupby('target_compnd')
Uw_best_total_delta_no_affi_df = grouped.apply(lambda x: x.sort_values(by=['Total_delta', 'Lig_delta', 'Core_RMSD'], ascending=[True, False, True]).head(1))
Uw_best_total_delta_no_affi_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_Uw_no_affi.csv', sep='\t', index=False)

## 5.2 merge -> `PLANet_Uw_affi_containing_NA.csv`
Uw_best_total_delta_no_affi_reset_idx_df = Uw_best_total_delta_no_affi_df.reset_index(drop=True).copy()
PLANet_Uw_affi_df = pd.merge(Uw_best_total_delta_no_affi_reset_idx_df, affi_df_for_ml, on='target_compnd', how="left")
PLANet_Uw_affi_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_Uw_affi_containing_NA.csv', sep='\t', index=False)

## 5.3 Remove NA for PLANet_Uw, and obtain `unique_identify` -> `PLANet_Uw_dealt.csv`
PLANet_Uw_affi_rm_NA_df = PLANet_Uw_affi_df[PLANet_Uw_affi_df['-logAffi'].notna()].copy()
PLANet_Uw_affi_rm_NA_df['unique_identify'] = [f'{row.Target_chembl_id}_{row.Cry_lig_name}_{row.Similar_compnd_name}' for row in PLANet_Uw_affi_rm_NA_df.itertuples()]
PLANet_Uw_affi_rm_NA_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_Uw_dealt.csv', sep='\t', index=False)

## 5.4 `PLANet_Uw_final.csv`
Uw_simple_df = PLANet_Uw_affi_rm_NA_df[['unique_identify', '-logAffi']]
Uw_simple_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_Uw_final.csv', sep = "\t", index = False)
grouped = Uw_simple_df.groupby('unique_identify')
Uw_median_df = grouped['-logAffi'].median().reset_index()
Uw_median_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_Uw_final_median.csv', sep = "\t", index = False)

## 5.5 PLANet_Uw_median_affinity_only -> add other information
Uw_best_total_delta_no_affi_reset_idx_df['unique_identify'] = [f'{row.Target_chembl_id}_{row.Cry_lig_name}_{row.Similar_compnd_name}' for row in Uw_best_total_delta_no_affi_reset_idx_df.itertuples()]
Uw_median_more_info_df = pd.merge(Uw_median_df, Uw_best_total_delta_no_affi_reset_idx_df, on='unique_identify', how="left")
Uw_median_more_info_df.to_csv(f'{wrkdir}/v2019_dataset/index/PLANet_Uw_dealt_median.csv', sep='\t', index=False)

# For SAR
affinity_only_final_df = pd.read_csv('target_compnd_affi_all_info_Corrected_manually_Rm_extrem_min_Rm_part_less_sign.csv', sep='\t')
PLANet_all_no_affi_df = pd.read_csv(f'{wrkdir}/v2019_dataset/index/PLANet_all_no_affi.csv', sep='\t')
PLANet_all_affi_for_SAR_df = pd.merge(PLANet_all_no_affi_df, affinity_only_final_df, on='target_compnd', how="left")
PLANet_all_affi_for_SAR_df.to_csv(f'{wrkdir}/v2019_dataset/index/For_SAR/PLANet_all_All_affi_for_SAR.csv', sep='\t', index=False)

PLANet_all_affi_for_SAR_rm_NA_df = PLANet_all_affi_for_SAR_df[PLANet_all_affi_for_SAR_df['-logAffi'].notna()].copy()
PLANet_all_affi_for_SAR_rm_NA_df['unique_identify'] = [f'{row.Target_chembl_id}_{row.Cry_lig_name}_{row.Similar_compnd_name}' for row in PLANet_all_affi_for_SAR_rm_NA_df.itertuples()]
PLANet_all_affi_for_SAR_rm_NA_df.to_csv(f'{wrkdir}/v2019_dataset/index/For_SAR/PLANet_all_for_SAR_rm_NA.csv', sep='\t', index=False) #254614

PLANet_Uw_no_affi_df = pd.read_csv(f'{wrkdir}/v2019_dataset/index/PLANet_Uw_no_affi.csv', sep='\t')
PLANet_Uw_affi_for_SAR_df = pd.merge(PLANet_Uw_no_affi_df, affinity_only_final_df, on='target_compnd', how="left")
PLANet_Uw_affi_for_SAR_df.to_csv(f'{wrkdir}/v2019_dataset/index/For_SAR/PLANet_Uw_All_affi_for_SAR.csv', sep='\t', index=False)

PLANet_Uw_affi_for_SAR_rm_NA_df = PLANet_Uw_affi_for_SAR_df[PLANet_Uw_affi_for_SAR_df['-logAffi'].notna()].copy()
PLANet_Uw_affi_for_SAR_rm_NA_df['unique_identify'] = [f'{row.Target_chembl_id}_{row.Cry_lig_name}_{row.Similar_compnd_name}' for row in PLANet_Uw_affi_for_SAR_rm_NA_df.itertuples()]
PLANet_Uw_affi_for_SAR_rm_NA_df.to_csv(f'{wrkdir}/v2019_dataset/index/For_SAR/PLANet_Uw_for_SAR_rm_NA.csv', sep='\t', index=False) #106072
