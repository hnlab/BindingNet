import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import oddt
from oddt import toolkit
from oddt.scoring.descriptors import close_contacts_descriptor, oddt_vina_descriptor

def extract_rf_v1_feature(ligand, protein, cutoff=6):
    ligand_atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    protein_atomic_nums = [6, 7, 8, 16]
    # cutoff = 12
    rfscore_descriptor_cc = close_contacts_descriptor(
        cutoff=cutoff,
        protein_types=protein_atomic_nums,
        ligand_types=ligand_atomic_nums)
    feature = rfscore_descriptor_cc.build(ligand, protein) #(1,36): cutoff内各pair的个数
    return rfscore_descriptor_cc.titles, feature

def extract_rf_v2_feature(ligand, protein, cutoff=6):
    ligand_atomic_nums = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    protein_atomic_nums = [6, 7, 8, 16]
    # cutoff = np.array([0, 2, 4, 6, 8, 10, 12])
    cutoff_list = np.arange(0,cutoff+1,2)
    descriptors = close_contacts_descriptor(
        cutoff=cutoff_list,
        protein_types=protein_atomic_nums,
        ligand_types=ligand_atomic_nums)
    feature = descriptors.build(ligand, protein) #(1, 36*6): 6种cutoff内各pair的个数(contact counts)
    return descriptors.titles, feature


def extract_rf_vina_feature(ligand, protein):
    vina_scores = ['vina_gauss1',
                'vina_gauss2',
                'vina_repulsion',
                'vina_hydrophobic',
                'vina_hydrogen',
                'vina_num_rotors']
    vina = oddt_vina_descriptor(vina_scores=vina_scores)
    feature = vina.build(ligand, protein) #(1,6): 6个具体的feature
    return vina_scores, feature


parser = argparse.ArgumentParser(
    description="extract feature for RFscore",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '-i',
    '--index_file',
    default='/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/1-remove_same_id_in_core_set/PLANet_Uw/PLANet_Uw_remove_core_set_ids.csv',
    help='index file: unique_identify, -logAffi',
)
parser.add_argument(
    '-s',
    '--structure_dir',
    default='/pubhome/xli02/project/PLIM/v2019_dataset',
    help='structure directory',
)
parser.add_argument(
    '-o',
    '--output_dir',
    default='/pubhome/xli02/project/PLIM/deep_learning/RFscore/featured_data/rm_core_ids/PLANet_Uw',
    help='output directory',
)
parser.add_argument(
    '-dn',
    '--dataset_name',
    default='PLANet_Uw',
    help='dataset name',
)
parser.add_argument(
    '-c',
    '--cutoff',
    default=6,
    type=int,
    help='distance cutoff of counted contact between atoms of protein and ligand',
)

args = parser.parse_args()
index_file = args.index_file
structure_dir = args.structure_dir
output_dir = args.output_dir
dataset_name = args.dataset_name
cutoff = args.cutoff

if not Path.exists(Path(output_dir)):
    Path.mkdir(Path(output_dir))

planet = pd.read_csv(index_file, sep="\t")

unique_identify_list=[]
aff_list=[]
for i in range(planet.shape[0]):
    name = planet.iloc[i,]['unique_identify']
    print(i)
    target = name.split('_')[0]
    pdbid = name.split('_')[1]
    compnd = name.split('_')[2]
    target_pdb_dir = f'{structure_dir}/web_client_{target}/{target}_{pdbid}'
    candi_lig = f'{target_pdb_dir}/{compnd}/compound.sdf'
    rec_pdb = f'{target_pdb_dir}/rec_opt/rec_h_opt.pdb'
    if not Path.exists(Path(candi_lig)):
        print(f"no ligand for {name} ({dataset_name} dataset)")
        continue
    unique_identify_list.append(name)
    aff_list.append(planet.iloc[i,]['-logAffi'])
    protein = next(oddt.toolkits.ob.readfile('pdb', rec_pdb))
    ligand = next(oddt.toolkits.ob.readfile('sdf', candi_lig))
    if i == 0:
        title_1, feature_1 = extract_rf_v1_feature(ligand, protein, cutoff)
        title_2, feature_2 = extract_rf_v2_feature(ligand, protein, cutoff)
        title_3, feature_3 = extract_rf_vina_feature(ligand, protein)
    else:
        _, feature_36 = extract_rf_v1_feature(ligand, protein, cutoff)
        _, feature_216 = extract_rf_v2_feature(ligand, protein, cutoff)
        _, feature_vina = extract_rf_vina_feature(ligand, protein)
        feature_1 = np.vstack((feature_1, feature_36)) #(n_lig, 36)
        feature_2 = np.vstack((feature_2, feature_216)) #(n_lig, 216)
        feature_3 = np.vstack((feature_3, feature_vina)) #(n_lig, 6)

features = np.hstack((feature_1, feature_2, feature_3))
titles = title_1 + title_2 + title_3
# print(titles)
# print(features)

data = pd.DataFrame(features, columns=titles)
data['unique_identify'] = unique_identify_list
data['-logAffi'] = aff_list

data.to_csv(f"{output_dir}/{dataset_name}_RFscore_feature_{cutoff}A.csv", sep='\t',index=False)

