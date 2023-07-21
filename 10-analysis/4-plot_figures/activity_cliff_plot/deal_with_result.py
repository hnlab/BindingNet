import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

wrkdir = '/pubhome/xli02/project/PLIM/activity_cliff'
successed_df = pd.read_csv(f'{wrkdir}/results/index/target_category.csv', sep='\t') # 455

all_targets_mmps_list = []
for row in successed_df.itertuples():
    target_id = row.target_chemb_id
    target_dir = f'{wrkdir}/results/{target_id}'
    mmp_all_df = pd.read_csv(f'{target_dir}/{target_id}_para_out_unique_trans_per_pair_with_cliffs_and_crytps.csv')
    mmps_all_list = np.sort(mmp_all_df[['mol1', 'mol2']], axis=1).tolist()
    mmps_all_one_way = [list(m) for m in set(tuple(mmp) for mmp in mmps_all_list)]
    successed_df.loc[row.Index, 'mmp_num'] = len(mmps_all_one_way)

    successed_df.loc[row.Index, 'mmp_cliffs_num'] = len(mmp_all_df[(mmp_all_df['mmp_cliff'])]) / 2
    successed_df.loc[row.Index, 'mmp_cliffs_with_crytps_num'] = len(mmp_all_df[(mmp_all_df['mmp_cliff']) & (mmp_all_df['crystal_ligand'].notna())]) / 2
    all_targets_mmps_list.append(mmp_all_df)
all_targets_mmps_df = pd.concat(all_targets_mmps_list, ignore_index=True)
all_targets_mmps_df.to_csv(f'{wrkdir}/results/index/all_targets_mmp.csv', sep='\t', index=False)
# len(all_targets_mmps_df) # 577048
# len(all_targets_mmps_df)/2 # 288524
# len(all_targets_mmps_df[all_targets_mmps_df['mmp_cliff']]) /2 # 52524
# len(all_targets_mmps_df[(all_targets_mmps_df['mmp_cliff']) & (all_targets_mmps_df['crystal_ligand'].notna())]) /2  # 51131


# all_targets_mmps_cliff_with_cry = all_targets_mmps_df[(all_targets_mmps_df['mmp_cliff']) & (all_targets_mmps_df['crystal_ligand'].notna())]
# len(set(list(all_targets_mmps_cliff_with_cry['mol1']) + list(all_targets_mmps_cliff_with_cry['mol2']))) # 25938

all_targets_mmps_df_trans_count = pd.DataFrame({'count':all_targets_mmps_df.groupby('transformation').size()}).reset_index().sort_values(by='count', ascending=False)
all_targets_mmps_df_trans_count_top = all_targets_mmps_df_trans_count.sort_values(by='count', ascending=False).head(40)
trans_count_one_way_list = []
one_way_trans = []
for trans in set(all_targets_mmps_df_trans_count_top['transformation']):
    trans_df = all_targets_mmps_df_trans_count_top[all_targets_mmps_df_trans_count_top['transformation'] == trans].copy()
    if f"{trans.split('>>')[1]}>>{trans.split('>>')[0]}" not in one_way_trans:
        one_way_trans.append(trans)
        trans_count_one_way_list.append(trans_df)
all_trans_count_one_way_df = pd.concat(trans_count_one_way_list, ignore_index=True).sort_values(by='count', ascending=False)
all_trans_count_one_way_df['trans_1'] = all_trans_count_one_way_df['transformation'].str.split('>>').str[0]
all_trans_count_one_way_df['trans_2'] = all_trans_count_one_way_df['transformation'].str.split('>>').str[1]
all_trans_count_one_way_df['trans'] = [f'{row.trans_1} <-> {row.trans_2}' for row in all_trans_count_one_way_df.itertuples()]
# all_trans_count_one_way_df.sort_values(by='count', ascending=False)
# all_trans_count_one_way_df
fig, ax= plt.subplots()
all_trans_count_one_way_df['count'].plot(kind='bar', color=['b'])
# ax.set_xticklabels(all_trans_count_one_way_df['trans'], fontsize=8)
ax.set_xticklabels(['H<->CH3', 'H<->F', 'H<->Cl', 'H<->OCH3', 'H<->OH', 'CH3<->CH2CH3', 'F<->Cl', 'H<->CH2CH3', 'H<->CF3', 'Cl-CH3', 'H<->Ph', 'H<->NH2', 'F-CH3', 'Cl<->OCH3', 'F<->OCH3', 'H<->CN', 'CH3<->CH(CH3)2', 'CH3<->OCH3', 'H<->Br', 'H<->CH(CH3)2'], fontsize=8)
fig.autofmt_xdate(rotation=45)
plt.ylabel('count', fontsize=12)
plt.xlabel('transormation', fontsize=12)
plt.title(f'Hot transformations for all MMPs', fontsize=15)
plt.savefig(f'all_targets_mmp_hot_trans_one_way.png', dpi=300, bbox_inches='tight')
plt.close()

all_targets_mmps_cliffs_with_crylig = all_targets_mmps_df[(all_targets_mmps_df['mmp_cliff']) & (all_targets_mmps_df['crystal_ligand'].notna())]
# len(all_targets_mmps_cliffs_with_crylig) # 102262
# len(set(list(all_targets_mmps_cliffs_with_crylig['mol1']) + list(all_targets_mmps_cliffs_with_crylig['mol2']))) # 25938
all_targets_mmps_cliffs_with_crylig.to_csv(f'{wrkdir}/results/index/all_targets_mmp_cliffs_with_crytps.csv', sep='\t', index=False)

# all_targets_mmps_cliffs_with_crylig_trans_count = pd.DataFrame({'count':all_targets_mmps_cliffs_with_crylig.groupby('transformation').size()}).reset_index().sort_values(by='count', ascending=False)
# all_targets_mmps_cliffs_with_crylig_trans_count['trans_1'] = all_targets_mmps_cliffs_with_crylig_trans_count['transformation'].str.split('>>').str[0]
# all_targets_mmps_cliffs_with_crylig_trans_count['trans_2'] = all_targets_mmps_cliffs_with_crylig_trans_count['transformation'].str.split('>>').str[1]

# trans_count_one_way_list = []
# one_way_trans = []
# for trans in set(all_targets_mmps_cliffs_with_crylig_trans_count['transformation']):
#     trans_df = all_targets_mmps_cliffs_with_crylig_trans_count[all_targets_mmps_cliffs_with_crylig_trans_count['transformation'] == trans].copy()
#     if f"{trans.split('>>')[1]}>>{trans.split('>>')[0]}" not in one_way_trans:
#         one_way_trans.append(trans)
#         trans_count_one_way_list.append(trans_df)

# trans_count_one_way_df = pd.concat(trans_count_one_way_list, ignore_index=True)
# trans_count_one_way_df['trans'] = [f'{row.trans_1} <-> {row.trans_2}' for row in trans_count_one_way_df.itertuples()]
# trans_count_one_way_df_top = trans_count_one_way_df.sort_values(by='count', ascending=False).head(20)

# fig, ax= plt.subplots()
# trans_count_one_way_df_top['count'].plot(kind='bar', color=['b'])
# # ax.set_xticklabels(trans_count_one_way_df_top['trans'], fontsize=8)
# ax.set_xticklabels(['H<->CH3', 'H<->Cl', 'H<->F', 'H<->OH', 'H<->OCH3', 'H<->Ph', 'H<->CH2CH3', 'H<->NH2', 'H<->CF3', 'H<->Br', 'H<->CH(CH3)2', 'F<->Cl', 'H<->COOH', 'H<->CN', 'CH3<->CH(CH3)2', 'CH3<->CH2CH3', 'CH3<->CF3', 'CH3<->Ph', 'F<->OCH3', 'H-CH2OH'], fontsize=8)
# fig.autofmt_xdate(rotation=45)
# plt.ylabel('count', fontsize=12)
# plt.xlabel('transormation', fontsize=12)
# plt.title(f'Hot transformations for MMP-cliffs with crystal templates', fontsize=15)
# plt.savefig(f'results/index/all_targets_mmp_cliffs_with_crytps_hot_trans_one_way.png', dpi=300, bbox_inches='tight')
# plt.close()

# sns.countplot(data=successed_df.sort_values(by='categories'), x="categories")
# plt.title(f'Categories of {len(successed_df)} targets with MMP-cliffs and crystal templates', fontsize=15)
# plt.ylabel('count', fontsize=12)
# plt.xlabel('categoties', fontsize=12)
# plt.savefig(f'results/index/target_category.png', dpi=300, bbox_inches='tight')
# plt.close()
