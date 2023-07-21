'''
Combine all feature files for PDBbind_minimized and PLANet_all: 'pdb_id' to 'unique_identify'
'''
import pandas as pd
from pathlib import Path

PDBbind_minimized_features = [str(p) for p in list(Path('/pubhome/xli02/project/PLIM/deep_learning/RFscore/featured_data/whole_set/PDBbind_minimized').glob('*csv'))]
PDBbind_minimized_features.sort()

PLANet_all_features = [str(p) for p in list(Path('/pubhome/xli02/project/PLIM/deep_learning/RFscore/featured_data/whole_set/PLANet_all').glob('*csv'))]
PLANet_all_features.sort()

pdbbind_df = pd.concat([pd.read_csv(f, sep='\t') for f in PDBbind_minimized_features], ignore_index=True)
pdbbind_df.rename(columns={'pdb_id':'unique_identify'}, inplace=True)

planet_df = pd.concat([pd.read_csv(f, sep='\t') for f in PLANet_all_features], ignore_index=True)
features_df = pd.concat([pdbbind_df, planet_df], ignore_index=True)
features_df.to_csv('/pubhome/xli02/project/PLIM/deep_learning/RFscore/featured_data/tow_datasets_features_6A.csv', sep='\t', index=False)
