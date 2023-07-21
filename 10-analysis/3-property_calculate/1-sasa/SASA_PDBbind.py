from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md
import subprocess
import time
from scipy.spatial import distance

PDBbind_whole = pd.read_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PDBbind_whole.csv', sep='\t')

skipped_mols = []
overlap_lig = []
overlap_rec = []
# logging.basicConfig(filename='/pubhome/xli02/project/PLIM/analysis/20220812_paper/SASA/SASA_calculate_PDBbind.log', filemode="w", level=print)

# error_list = ['1jik','5jim','2vf6','2nsl','5mys','4mdq','4bps','2ggd','4aq4','1qk3','2x7u','5myn','5chk','4ayr','1mns','4ayp','4bs0']

with open('/pubhome/xli02/project/PLIM/analysis/20220812_paper/SASA/PDBbind_whole_sasa.csv', 'w') as f:
    f.write(f'unique_identity,lig_sasa,rec_sasa,com_sasa,del_sasa\n')
    for row in PDBbind_whole.itertuples():
        uniq_id = row.pdb_id
        print(f'[{time.ctime()}] Start {uniq_id}:')
        # lig_file = f'{pdbbind_dir}/general_structure_only/{uniq_id}/{uniq_id}_ligand.smi'
        lig_file = f'/pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019/{uniq_id}/cry_lig_opt_converted.sdf'
        # complex_ = f'/pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019/{uniq_id}/cry_com.pdb'
        rec = f'/pubhome/xli02/project/PLIM/v2019_dataset/PDBbind_v2019/{uniq_id}/rec_h_opt.pdb'

        lig_ob_convert = f'/tmp/xli/{uniq_id}/lig_ob.pdb'
        lig_grepped = f'/tmp/xli/{uniq_id}/lig_ob_grepped.pdb'
        rec_grepped = f'/tmp/xli/{uniq_id}/rec.pdb'
        com = f'/tmp/xli/{uniq_id}/com.pdb'
        a_sb = subprocess.run(f'mkdir -p /tmp/xli/{uniq_id}; grep ^ATOM {rec} > {rec_grepped}; obabel {lig_file} -O{lig_ob_convert}; sed -i s/"ATOM  "/"HETATM"/g {lig_ob_convert}; grep ^HETATM {lig_ob_convert} > {lig_grepped}; cat {rec_grepped} {lig_grepped} > {com}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # logging.debug("\n" + a_sb.stdout.decode())
        # logging.debug("\n" + a_sb.stderr.decode())


        if not Path(lig_file).exists():
            print(f'{uniq_id} of PDBbind not exsits, skipped.')
            skipped_mols.append(uniq_id)
            continue
        try:
            lig_load = md.load(lig_grepped) #
            rec_load = md.load(rec_grepped)
            com_load = md.load(com)
            dist_lig = distance.cdist(lig_load.xyz[0], lig_load.xyz[0], "euclidean",)
            np.fill_diagonal(dist_lig, 1)
            if np.min(dist_lig) < pow(10,-4):
                overlap_lig.append(uniq_id)
                print(f'{uniq_id}_lig of PDBbind has overlap atoms: {np.argwhere(dist_lig<pow(10,-4))}')
                continue

            dist_rec = distance.cdist(rec_load.xyz[0], rec_load.xyz[0], "euclidean",)
            np.fill_diagonal(dist_rec, 1)
            if np.min(dist_rec) < pow(10,-4):
                overlap_rec.append(uniq_id)
                print(f'{uniq_id}_rec of PDBbind has overlap atoms: {np.argwhere(dist_rec<pow(10,-4))}')
                continue
        except:
            print(f'{uniq_id} of PDBbind load error, skipped.')
            skipped_mols.append(uniq_id)
            continue
        try:
            lig_sasa = np.sum(md.shrake_rupley(lig_load))
            rec_sasa = np.sum(md.shrake_rupley(rec_load))
            com_sasa = np.sum(md.shrake_rupley(com_load))
            del_sasa = lig_sasa + rec_sasa - com_sasa
            PDBbind_whole.loc[row.Index, 'lig_sasa'] = lig_sasa
            PDBbind_whole.loc[row.Index, 'rec_sasa'] = rec_sasa
            PDBbind_whole.loc[row.Index, 'complex_sasa'] = com_sasa
            PDBbind_whole.loc[row.Index, 'delta_sasa'] = del_sasa
            f.write(f'{uniq_id},{lig_sasa},{rec_sasa},{com_sasa},{del_sasa}\n')
            b_sb = subprocess.run(f'rm -r /tmp/xli/{uniq_id}', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        except:
            print(f'{uniq_id} of PDBbind calculate SASA error, skipped.')
            skipped_mols.append(uniq_id)
            
# PDBbind_whole.to_csv('/pubhome/xli02/project/PLIM/analysis/20220812_paper/distribution/property/PDBbind_whole_sasa.csv', sep='\t', index=False)
print(f'skipped mols:{len(skipped_mols)}')
print(skipped_mols)
print(f'overlap ligs: {len(overlap_lig)}')
print(overlap_lig)
print(f'overlap recs: {len(overlap_rec)}')
print(overlap_rec)
