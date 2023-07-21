'''
if RMSD < 2 A, add it into a list
'''
import pandas as pd
from pathlib import Path

# index_file = '/home/xli/dataset/PDBbind_v2019/raw/plain-text-index/index/INDEX_general_PL_data_grepped.2019'
index_file = '/home/xli/dataset/PDBbind_v2019/raw/plain-text-index/index/grepped/INDEX_general_PL_data_grepped.2019'
index_df = pd.read_csv(index_file, sep = ' ', header=None, names=['pdb_id', '-logAffi'])

wrkdir = '/home/xli/Documents/projects/ChEMBL-scaffold/v2019_dataset/PDBbind_v2019'

for row in index_df.itertuples():
    extrac_log = f'{wrkdir}/{row.pdb_id}/extract_pocket_6A.log'
    if Path(extrac_log).exists():
        with open(extrac_log, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'RMSD between' in line:
                index_df.loc[row.Index, 'RMSD'] = float(line.split(' ')[6])

index_with_RMSD_filtered_df = index_df[index_df['RMSD'] < 2]
index_with_RMSD_filtered_df.to_csv(f'{wrkdir}/index/INDEX_general_PL_data_filtered.2019', sep = "\t", index = False)

dealt_df = index_with_RMSD_filtered_df[['pdb_id', '-logAffi']]
dealt_df.to_csv(f'{wrkdir}/index/INDEX_general_PL_data_filtered_dealt.2019', sep = "\t", index = False)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

sns.displot(dealt_df, x="-logAffi")
plt.xlim(min(dealt_df['-logAffi']),max(dealt_df['-logAffi']))
plt.title(f'The distribution of binding affinity in PDBbind(N={len(dealt_df)})')
plt.savefig(f'{wrkdir}/index/general_PL_affinity_distribution.png')

sns.displot(index_with_RMSD_filtered_df, x="RMSD")
plt.title(f'The distribution of RMSD in PDBbind_v2019(N={len(dealt_df)})')
plt.savefig(f'{wrkdir}/index/general_PL_RMSD_distribution.png')
