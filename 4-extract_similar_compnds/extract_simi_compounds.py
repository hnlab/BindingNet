"""
script for Calculating similarity among candidate compounds and crystal ligands
And extract compounds with similarity > 0.7
"""
import csv
import time
import argparse
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit import DataStructs


def chembl2pdbid(target_file):
    chembl2pdb = {}
    header = True
    with open(target_file, "r") as target_file:
        lines = target_file.readlines()
    for pdb_ids, chembl_ids in [
        (line.split("\t")[0], line.rstrip().split("\t")[2][:-1]) for line in lines
    ]:
        if header:
            header = False
            continue
        for chembl_id in chembl_ids.split(";"):
            if chembl_id in chembl2pdb:
                for pdb_id in pdb_ids.split(","):
                    if pdb_id in chembl2pdb[chembl_id]:
                        continue
                    else:
                        chembl2pdb[chembl_id].append(pdb_id)
            else:
                chembl2pdb[chembl_id] = pdb_ids.split(",")
    return chembl2pdb


def obtain_ref_mol_and_source(pdbids):
    pdbbind_dir = '/home/xli/dataset/PDBbind_v2019/general_structure_only'
    pdbbind_fixed_sdf_dir = '/home/xli/dataset/PDBBind_v2019_general_fixed_sdf'
    reference_mols =[]
    reference_mol_source = []
    for pdb_id in pdbids:
        cry_lig_fixed_sdf = f'{pdbbind_fixed_sdf_dir}/{pdb_id}_ligand.fixed.sdf'
        cry_lig_sdf = f'{pdbbind_dir}/{pdb_id}/{pdb_id}_ligand.sdf'
        cry_lig_mol2 = f'{pdbbind_dir}/{pdb_id}/{pdb_id}_ligand.mol2'
        cry_lig_smi = f'{pdbbind_dir}/{pdb_id}/{pdb_id}_ligand.smi'
        if Path(cry_lig_smi).exists() == False:
            print(f"[{time.ctime()}] Structure of {pdb_id} do not exist, skipped.")
            continue
        elif Chem.SmilesMolSupplier(cry_lig_smi, delimiter='\t',titleLine=False)[0] is not None:
            reference_mol = Chem.SmilesMolSupplier(cry_lig_smi, delimiter='\t',titleLine=False)[0]
            cry_file = 'smi'
            print(f"[{time.ctime()}] Use {pdb_id}_ligand.smi as reference_mol. But it cannot be used to align.")
        elif Chem.MolFromMol2File(cry_lig_mol2) is not None:
            reference_mol = Chem.MolFromMol2File(cry_lig_mol2)
            cry_file = 'mol2'
            print(f"[{time.ctime()}] Use {pdb_id}_ligand.mol2 as reference_mol.")
        elif Chem.SDMolSupplier(cry_lig_sdf)[0] is not None:
            reference_mol = Chem.SDMolSupplier(cry_lig_sdf)[0]
            cry_file = 'sdf'
            print(f"[{time.ctime()}] Use {pdb_id}_ligand.sdf as reference_mol.")
        elif Path(cry_lig_fixed_sdf).exists() and Chem.SDMolSupplier(cry_lig_fixed_sdf)[0] is not None:
            reference_mol = Chem.SDMolSupplier(cry_lig_fixed_sdf)[0]
            cry_file = 'fixed_sdf'
            print(f"[{time.ctime()}] Use {pdb_id}_ligand.fixed.sdf as reference_mol.")
        else:
            print(f"[{time.ctime()}] Template {pdb_id}_ligand(.fixed).sdf/mol2/smi ALL cannot be read by rdkit, skipped.")
            continue
        reference_mols.append(reference_mol)
        reference_mol_source.append(cry_file)
    return reference_mols, reference_mol_source


def getfps(mols, fp_type="basic"):
    fps = []
    for mol in mols:
        if mol is not None:
            if fp_type == "basic":
                fps.append(FingerprintMols.FingerprintMol(mol))
            elif fp_type == "maccs":
                fps.append(MACCSkeys.GenMACCSKeys(mol))
            elif fp_type == "ap":
                fps.append(Pairs.GetAtomPairFingerprintAsBitVect(mol))
            elif fp_type == "tt":
                fps.append(Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol))
            elif fp_type == "morgan":
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
        else:
            fps.append(0)   #如果分子被rdkit读入为"None"，其分子指纹为0
    return fps


def similarity_filter_write(reference_mols, candi_mols, reference_mol_source, fp_type, sim_cutoff = 0.7):    #提取相似性>cutoff的compounds，并写入文件
    columns = ["cry_lig_name",
               "cry_lig_smiles",
               "similar_compounds_name",
               "similar_compounds_smiles",
               "similarity",
               "reference_mol_source",
               ]
    simi_df = pd.DataFrame(columns=columns)

    refer_fps = getfps(reference_mols, fp_type = fp_type)
    candi_fps = getfps(candi_mols, fp_type = fp_type)
    for i in range(len(refer_fps)):
        candi_names = []
        for j in range(len(candi_fps)):
            if candi_fps[j] == 0:
                print(f"Candidate ligand {candi_mols[j]}.smi from ChEMBL cannot be read by rdkit, skipped.")
                continue
            else:
                simi = DataStructs.FingerprintSimilarity(refer_fps[i], candi_fps[j])
                if simi > sim_cutoff:                                  #不去除CHEMBL中与crystal_lig相似性为1的候选分子
                    temp_name = reference_mols[i].GetProp('_Name').split('_')[0].lower()         #部分template转换为smi后，名称大写，如"3P3J"等
                    candi_name = candi_mols[j].GetProp('_Name')
                    if candi_name not in candi_names:                                      #去重
                        simi_df = simi_df.append(
                            pd.DataFrame(
                                [[
                                    temp_name, Chem.MolToSmiles(reference_mols[i]),
                                    candi_name, Chem.MolToSmiles(candi_mols[j]),
                                    simi, reference_mol_source[i]
                                ]], 
                                columns=columns
                            ),
                            ignore_index=True
                        )
                        candi_names.append(candi_name)

    simi_file = str(WRKDIR/f'v2019_dataset/web_client_{target_chembl_id}/simi_{sim_cutoff}_fp_{fp_type}.csv')
    simi_df.to_csv(simi_file, sep = "\t", index = False)            #写入csv文件
    len_cry_lig = len(set(simi_df['cry_lig_name']))
    len_unique_compounds = len(set(simi_df['similar_compounds_name']))
    print(
        f"[{time.ctime()}] Calculated similarity. "
        f"Obtained {len_unique_compounds} unique compounds similar to "
        f"the {len_cry_lig}/{len(chembl2pdb[target_chembl_id])} crystal ligands of target {target_chembl_id}. "
    )


def calculate_similarity(target_chembl_id, sim_cutoff = 0.7, fp_type = "basic"):
    candi_file = str(WRKDIR/f'v2019_dataset/web_client_{target_chembl_id}/web_client_{target_chembl_id}-smiles-chembl_id.smi')
    candi_mols = Chem.SmilesMolSupplier(candi_file, delimiter='\t')
    if len(candi_mols) == 0:
        print(f"[{time.ctime()}] There is no candidate compounds for {target_chembl_id}, exit.")
        return

    pdbids = chembl2pdb[target_chembl_id]
    reference_mols, reference_mol_source = obtain_ref_mol_and_source(pdbids)
    print(f'[{time.ctime()}] {len(reference_mols)} / {len(pdbids)} crystal ligands can be read by rdkit.')

    similarity_filter_write(reference_mols, candi_mols, reference_mol_source, fp_type = fp_type, sim_cutoff = sim_cutoff)


parser = argparse.ArgumentParser(
    description="Extracting similar compounds with crystal ligands for each target_ChEMBL_ID",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '-s',
    '--sim',
    default=0.7,
    help='similarity cutoff among candidate ligands and crystal ligands',
)
parser.add_argument(
    '-f',
    '--fp_type',
    default='basic',
    choices=['basic', 'maccs', 'ap', 'tt', 'morgan'],
    help='the method of calculating fingerprint of compounds',
)
parser.add_argument(
    '-t',
    '--target_chembl_id',
    default='CHEMBL301',
    help='CHEMBL id of target',
)

WRKDIR = Path("/home/xli/Documents/projects/ChEMBL-scaffold")
target_file = WRKDIR/'pipeline/2-query_target_ChEMBLid/converted_PDBIDs_INDEX_general_PL_data.2019.tab.tsv'
chembl2pdb = chembl2pdbid(target_file)

args = parser.parse_args()
target_chembl_id = args.target_chembl_id
sim_cutoff = args.sim
fp_type = args.fp_type

print(f"[{time.ctime()}] Calculating similarity among crystal ligands and candidate compounds of {target_chembl_id}. ")
calculate_similarity(target_chembl_id, sim_cutoff=sim_cutoff, fp_type = fp_type)