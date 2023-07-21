from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem
from rdkit import DataStructs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mdtraj as md
import subprocess
import logging
import time
from scipy.spatial import distance

PLANet_df=pd.read_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PLANet_property.csv', sep='\t')

# logging.basicConfig(filename='/pubhome/xli02/project/PLIM/analysis/20220812_paper/SASA/SASA_calculate_PLANet.log', filemode="w", level=logging.DEBUG)

skipped_mols = []
overlap_lig = []
overlap_rec = []

with open('/pubhome/xli02/project/PLIM/analysis/20220812_paper/SASA/PLANet_property_sasa.csv', 'w') as f:
    f.write(f'unique_identity,lig_sasa,rec_sasa,com_sasa,del_sasa\n')
    for row in PLANet_df.itertuples():
        print(f'[{time.ctime()}] Start {row.unique_identify}:')
        lig = f'/pubhome/xli02/project/PLIM/v2019_dataset/web_client_{row.Target_chembl_id}/{row.Target_chembl_id}_{row.Cry_lig_name}/{row.Similar_compnd_name}/{row.unique_identify}_dlig_-20_dtotal_100_CoreRMSD_2.0_final.pdb'
        rec = f'/pubhome/xli02/project/PLIM/v2019_dataset/web_client_{row.Target_chembl_id}/{row.Target_chembl_id}_{row.Cry_lig_name}/rec_opt/rec_h_opt.pdb'

        a_sb = subprocess.run(f'mkdir -p /tmp/xli/{row.unique_identify}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # logging.debug("\n" + a_sb.stdout.decode())
        # logging.debug("\n" + a_sb.stderr.decode())
        
        rec_ob_convert = f'/tmp/xli/{row.unique_identify}/rec.pdb'
        rec_ob_grep = f'/tmp/xli/{row.unique_identify}/rec_grepped.pdb'
        lig_grep = f'/tmp/xli/{row.unique_identify}/lig_grepped.pdb'
        complex_ = f'/tmp/xli/{row.unique_identify}/com.pdb'

        b_sb = subprocess.run(f'obabel {rec} -O{rec_ob_convert}; grep "ATOM" {rec_ob_convert} > {rec_ob_grep}; grep "ATOM" {lig} > {lig_grep}; cat {rec_ob_grep} {lig_grep} > {complex_}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # c_sb = subprocess.run(f'cat {rec_ob_convert} {lig} > {complex_}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # logging.debug("\n" + b_sb.stdout.decode())
        # logging.debug("\n" + b_sb.stderr.decode())

        try:
            lig_load = md.load(lig)
            rec_load = md.load(rec_ob_convert)
            com_load = md.load(complex_)
            dist_lig = distance.cdist(lig_load.xyz[0], lig_load.xyz[0], "euclidean",)
            np.fill_diagonal(dist_lig, 1)
            if np.min(dist_lig) < pow(10,-4):
                overlap_lig.append(row.unique_identify)
                print(f'{row.unique_identify}_lig of PLANet has overlap atoms: {np.argwhere(dist_lig<pow(10,-4))}')
                continue

            dist_rec = distance.cdist(rec_load.xyz[0], rec_load.xyz[0], "euclidean",)
            np.fill_diagonal(dist_rec, 1)
            if np.min(dist_rec) < pow(10,-4):
                overlap_rec.append(row.unique_identify)
                print(f'{row.unique_identify}_rec of PLANet has overlap atoms: {np.argwhere(dist_rec<pow(10,-4))}')
                continue
        except:
            print(f'{row.unique_identify} of PLANet load error, skipped.')
            skipped_mols.append(row.unique_identify)
            continue

        try:
            lig_sasa = np.sum(md.shrake_rupley(lig_load))
            rec_sasa = np.sum(md.shrake_rupley(rec_load))
            com_sasa = np.sum(md.shrake_rupley(com_load))
            del_sasa = lig_sasa + rec_sasa - com_sasa
            # PLANet_df.loc[row.Index, 'lig_sasa'] = lig_sasa
            # PLANet_df.loc[row.Index, 'rec_sasa'] = rec_sasa
            # PLANet_df.loc[row.Index, 'complex_sasa'] = com_sasa
            # PLANet_df.loc[row.Index, 'delta_sasa'] = del_sasa
            f.write(f'{row.unique_identify},{lig_sasa},{rec_sasa},{com_sasa},{del_sasa}\n')
            b_sb = subprocess.run(f'rm -r /tmp/xli/{row.unique_identify}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        except:
            logging.debug(f'{row.unique_identify} of PLANet calculate SASA error, skipped.')
            skipped_mols.append(row.unique_identify)

# PLANet_df.to_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PLANet_property_sasa.csv', sep='\t', index=False)
print(f'skipped mols:{len(skipped_mols)}')
print(skipped_mols)
print(f'overlap ligs: {len(overlap_lig)}')
print(overlap_lig)
print(f'overlap recs: {len(overlap_rec)}')
print(overlap_rec)

