import pandas as pd

wrkdir = '/pubhome/xli02/project/PLIM/activity_cliff'
successed_df = pd.read_csv(f'{wrkdir}/results/index/target_category_statistcs.csv', sep='\t')
# len(successed_df) # 455
# len(successed_df[successed_df['categories'] == 'CAT1']) # 163
# len(successed_df[successed_df['categories'] == 'CAT2']) # 253
# len(successed_df[successed_df['categories'] == 'CAT3']) # 39

# 1. CAT1
target_id = 'CHEMBL248'
target_dir = f'{wrkdir}/results/{target_id}'
cat1_mmp_compounds_pAffi_medianed = pd.read_csv(f'{target_dir}/{target_id}_mmp_cpds_pAffi_medianed.csv')
cat1_mmp_compounds_pAffi_medianed['target'] = 'CHEMBL248\n(Category 1)'
# len(cat1_mmp_compounds_pAffi_medianed) # 666

target_id = 'CHEMBL4822'
target_dir = f'{wrkdir}/results/{target_id}'
cat2_mmp_compounds_pAffi_medianed = pd.read_csv(f'{target_dir}/{target_id}_mmp_cpds_pAffi_medianed.csv')
cat2_mmp_compounds_pAffi_medianed['target'] = 'CHEMBL4822\n(Category 2)'
# len(cat2_mmp_compounds_pAffi_medianed) # 2600

target_id = 'CHEMBL204'
target_dir = f'{wrkdir}/results/{target_id}'
cat3_mmp_compounds_pAffi_medianed = pd.read_csv(f'{target_dir}/{target_id}_mmp_cpds_pAffi_medianed.csv')
cat3_mmp_compounds_pAffi_medianed['target'] = 'CHEMBL204\n(Category 3)'
# len(cat3_mmp_compounds_pAffi_medianed) # 710

test_df = pd.concat([cat1_mmp_compounds_pAffi_medianed, cat2_mmp_compounds_pAffi_medianed, cat3_mmp_compounds_pAffi_medianed])
cat1_mmp_compounds_pAffi_medianed['mmp_pAffi'].quantile([0.25, 0.75])
cat2_mmp_compounds_pAffi_medianed['mmp_pAffi'].quantile([0.25, 0.75])
cat3_mmp_compounds_pAffi_medianed['mmp_pAffi'].quantile([0.25, 0.75])

from matplotlib import pyplot as plt
import seaborn as sns

fig, ax= plt.subplots(figsize=(7,4))
ax.set_title(f'pAffi of MMP-forming compounds in 3 exemplary targets from each category', fontsize=15)
sns.boxplot(data=test_df, y="mmp_pAffi", x='target')
plt.ylabel('Experimental pAffi', fontsize=15)
plt.xlabel('Target(Categories)', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.savefig(f'three_targets_mmp_cpds_pAffi_medianed_v2.png', dpi=300, bbox_inches='tight')
plt.close()


fig, ax= plt.subplots(figsize=(7,5))
sns.countplot(data=successed_df.sort_values(by='categories'), x="categories")
ax.set_xticklabels(['Category 1', 'Category 2', 'Category 3'])
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
plt.title(f'Categories of {len(successed_df)} targets with MMP-cliffs and crystal templates', fontsize=15)
plt.ylabel('count', fontsize=15)
plt.xlabel('categoties', fontsize=15)
plt.savefig(f'target_category_v2.png', dpi=300, bbox_inches='tight')
plt.close()


all_targets_mmps_cliffs_with_crylig = pd.read_csv(f'{wrkdir}/results/index/all_targets_mmp_cliffs_with_crytps.csv', sep='\t')
all_targets_mmps_cliffs_with_crylig_trans_count = pd.DataFrame({'count':all_targets_mmps_cliffs_with_crylig.groupby('transformation').size()}).reset_index().sort_values(by='count', ascending=False)
all_targets_mmps_cliffs_with_crylig_trans_count['trans_1'] = all_targets_mmps_cliffs_with_crylig_trans_count['transformation'].str.split('>>').str[0]
all_targets_mmps_cliffs_with_crylig_trans_count['trans_2'] = all_targets_mmps_cliffs_with_crylig_trans_count['transformation'].str.split('>>').str[1]

trans_count_one_way_list = []
one_way_trans = []
for trans in set(all_targets_mmps_cliffs_with_crylig_trans_count['transformation']):
    trans_df = all_targets_mmps_cliffs_with_crylig_trans_count[all_targets_mmps_cliffs_with_crylig_trans_count['transformation'] == trans].copy()
    if f"{trans.split('>>')[1]}>>{trans.split('>>')[0]}" not in one_way_trans:
        one_way_trans.append(trans)
        trans_count_one_way_list.append(trans_df)

trans_count_one_way_df = pd.concat(trans_count_one_way_list, ignore_index=True)
trans_count_one_way_df['trans'] = [f'{row.trans_1} <-> {row.trans_2}' for row in trans_count_one_way_df.itertuples()]
trans_count_one_way_df_top = trans_count_one_way_df.sort_values(by='count', ascending=False).head(20)

fig, ax= plt.subplots(figsize=(7,4))
trans_count_one_way_df_top['count'].plot(kind='bar', color=['b'])
# ax.set_xticklabels(trans_count_one_way_df_top['trans'], fontsize=8)
ax.set_xticklabels(['H<->CH3', 'H<->Cl', 'H<->F', 'H<->OH', 'H<->OCH3', 'H<->Ph', 'H<->CH2CH3', 'H<->NH2', 'H<->CF3', 'H<->Br', 'H<->CH(CH3)2', 'F<->Cl', 'H<->COOH', 'H<->CN', 'CH3<->CH(CH3)2', 'CH3<->CH2CH3', 'CH3<->CF3', 'CH3<->Ph', 'F<->OCH3', 'H-CH2OH'], fontsize=12)
# ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
fig.autofmt_xdate(rotation=45)
plt.ylabel('count', fontsize=15)
plt.xlabel('transormation', fontsize=15)
plt.title(f'Hot transformations for MMP-cliffs with crystal templates', fontsize=15)
plt.savefig(f'all_targets_mmp_cliffs_with_crytps_hot_trans_one_way_v2.png', dpi=300, bbox_inches='tight')
plt.close()
