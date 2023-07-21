'''
For PLANet, remove compounds with same pdbids and similarity=1 with cry_lig
'''
import pandas as pd
from pathlib import Path

wrkdir = '/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/1-remove_same_id_in_core_set'

pdbbind_dir = '/pubhome/xli02/Downloads/dataset/PDBbind'
core_pdbid_file = f'{pdbbind_dir}/CASF-2016/core_pdbid.csv'
with open(core_pdbid_file, 'r') as f:
    lines = f.readlines()
core_pdbids = [line.rstrip() for line in lines]

# Uw
PLANet_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PLANet_Uw_dir = f'{wrkdir}/PLANet_Uw'
if not Path(PLANet_Uw_dir).exists():
    Path(PLANet_Uw_dir).mkdir()
PLANet_Uw_dealt_f = f'{PLANet_dir}/PLANet_Uw_dealt_median.csv'
PLANet_Uw_dealt_df = pd.read_csv(PLANet_Uw_dealt_f, sep='\t')
PLANet_Uw_rm_core_ids_df = PLANet_Uw_dealt_df[~((PLANet_Uw_dealt_df['Cry_lig_name'].isin(core_pdbids)) & (PLANet_Uw_dealt_df['Similarity'] == 1))]
PLANet_Uw_final_rm_core_ids_df = PLANet_Uw_rm_core_ids_df[['unique_identify', '-logAffi']]
PLANet_Uw_final_rm_core_ids_df.to_csv(f'{PLANet_Uw_dir}/PLANet_Uw_remove_core_set_ids.csv', sep='\t', index=False)

# all
PLANet_all_dir = f'{wrkdir}/PLANet_all'
if not Path(PLANet_all_dir).exists():
    Path(PLANet_all_dir).mkdir()
PLANet_all_dealt_f = f'{PLANet_dir}/PLANet_all_dealt_median.csv'
PLANet_all_dealt_df = pd.read_csv(PLANet_all_dealt_f, sep='\t')
PLANet_all_rm_core_ids_df = PLANet_all_dealt_df[~((PLANet_all_dealt_df['Cry_lig_name'].isin(core_pdbids)) & (PLANet_all_dealt_df['Similarity'] == 1))]
PLANet_all_final_rm_core_ids_df = PLANet_all_rm_core_ids_df[['unique_identify', '-logAffi']]
PLANet_all_final_rm_core_ids_df.to_csv(f'{PLANet_all_dir}/PLANet_all_remove_core_set_ids.csv', sep='\t', index=False)

# PDBbind_minimized
PDBbind_v19_minimized_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019/index'
PDBbind_minimized_f = f'{PDBbind_v19_minimized_dir}/PDBbind_v19_minimized_succeed_manually_modified_final.csv'
PDBbind_minimized_df = pd.read_csv(PDBbind_minimized_f, sep='\t')
PDBbind_v19_minimized_remove_core_set_df = PDBbind_minimized_df[~PDBbind_minimized_df['pdb_id'].isin(core_pdbids)]
PDBbind_minimized_dir = f'{wrkdir}/PDBbind_minimized'
if not Path(PDBbind_minimized_dir).exists():
    Path(PDBbind_minimized_dir).mkdir()
PDBbind_v19_minimized_remove_core_set_df.to_csv(f'{PDBbind_minimized_dir}/PDBbind_minimized_rm_core_ids.csv', sep='\t', index=False)

# PDBbind_minimized_intersected_Uw
PDBbind_minimized_intersected_Uw_f = f'{PDBbind_v19_minimized_dir}/PDBbind_minimized_intersected_Uw/PDBbind_minimized_final_intersected_PLANet_Uw.csv'
PDBbind_minimized_intersected_Uw_df = pd.read_csv(PDBbind_minimized_intersected_Uw_f, sep='\t')
PDBbind_v19_minimized_intersected_Uw_remove_core_set_df = PDBbind_minimized_intersected_Uw_df[~PDBbind_minimized_intersected_Uw_df['pdb_id'].isin(core_pdbids)]
PDBbind_minimized_intersected_Uw_dir = f'{wrkdir}/PDBbind_minimized_intersected_Uw'
if not Path(PDBbind_minimized_intersected_Uw_dir).exists():
    Path(PDBbind_minimized_intersected_Uw_dir).mkdir()
PDBbind_v19_minimized_intersected_Uw_remove_core_set_df.to_csv(f'{PDBbind_minimized_intersected_Uw_dir}/PDBbind_minimized_intersected_Uw_rm_core_ids.csv', sep='\t', index=False)

# # PDBbind_intersected_Uw
# PDBbind_intersected_Uw_dir = f'{wrkdir}/PDBbind_intersected_Uw'
# if not Path(PDBbind_intersected_Uw_dir).exists():
#     Path(PDBbind_intersected_Uw_dir).mkdir()
# PDBbind_intersected_Uw_df = pd.read_csv(f'{PLANet_dir}/PDBbind_intersection/PDBbind_intersected_PLANet_Uw.csv', sep='\t')
# PDBbind_intersected_Uw_remove_core_set_df = PDBbind_intersected_Uw_df[~PDBbind_intersected_Uw_df['pdb_id'].isin(core_pdbids)]
# PDBbind_intersected_Uw_remove_core_set_df.to_csv(f'{PDBbind_intersected_Uw_dir}/PDBbind_intersected_Uw_rm_core_ids.csv', sep='\t', index=False)
