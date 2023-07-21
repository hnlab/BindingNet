'''
For candidate compounds similar with crystal ligand(simi_cutoff > 0.7),
Align candidate compounds constraintly according to their max common substructure
 with crystal ligand, respectively
Sample according to the num of rotatable bonds:
 if rota_bonds_num <0 or >1000, sample 6 or 1000 times
'''
from pathlib import Path
import pandas as pd
import logging
import time

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolDescriptors
from rdkit.Chem.AllChem import AlignMol
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.SaltRemover import SaltRemover

from oddt.toolkits.extras.rdkit import AtomListToSubMol


def calcu_sample_num(reference_mol, cry_match, simi_mol):
    #generate core
    amap = list(cry_match)
    core = AtomListToSubMol(reference_mol, amap=amap, includeConformer=True)
    Chem.GetSymmSSSR(core)   # do ring perception
    core_rota_bonds_num = rdMolDescriptors.CalcNumRotatableBonds(core)
    simi_rota_bonds_num = rdMolDescriptors.CalcNumRotatableBonds(simi_mol)
    rota_bonds_num = simi_rota_bonds_num - core_rota_bonds_num
    if rota_bonds_num < 1:
        rota_bonds_num = 1
    if 6**rota_bonds_num > 1000:
        sample_num = 1000
    else:
        sample_num = 6**rota_bonds_num
    return sample_num


def add_fixed_point_align(mol, reference_mol, random_index, match, simi_mol_name, logger):
    code_1 = AllChem.EmbedMolecule(mol, randomSeed=random_index+1)
    if code_1 != 0:
        logger.warning(f"[{time.ctime()}] For {simi_mol_name}, " \
            f"'ff_mol.AddFixedPoint' also failed, skipped.")
        return code_1, None
    ref_conf = reference_mol.GetConformer()
    candi_conf = mol.GetConformer()
    ff_mol = AllChem.UFFGetMoleculeForceField(mol)
    for i, j in match:
        ff_mol.AddFixedPoint(i)
        candi_conf.SetAtomPosition(i, ref_conf.GetAtomPosition(j))
    ff_mol.Minimize(maxIts=200000)
    mol_3d = Chem.RemoveHs(mol)
    mol_3d_addh = Chem.AddHs(mol_3d, addCoords=True)
    return code_1, mol_3d_addh


def constrained_align_sample_minimize(simi_mol, reference_mol, mcs_mol, simi_mol_name, logger):
    cry_match = reference_mol.GetSubstructMatch(mcs_mol)
    candi_matches = simi_mol.GetSubstructMatches(mcs_mol)
    
    sample_num = calcu_sample_num(reference_mol, cry_match, simi_mol)

    #generate constrainted conformation
    mols = []
    for match_idx,candi_match in enumerate(candi_matches):
        match = list(zip(candi_match, cry_match))
        cmap = {pi: reference_mol.GetConformer().GetAtomPosition(xi) 
            for pi, xi in match}

        for random_index in range(sample_num):
            conf_idx = match_idx * sample_num + random_index
            mol = Chem.Mol(simi_mol)
            mol = Chem.AddHs(mol)
            code = AllChem.EmbedMolecule(mol, coordMap=cmap, randomSeed=random_index+1)   # 0和1的结果相同？
            if code != 0:   #whether converged
                logger.warning(f"[{time.ctime()}] {simi_mol_name}_{match_idx}_{random_index} EmbedMolecule failed, " \
                    f"Try 'ff_mol.AddFixedPoint'.")
                code_1, mol_3d_addh = add_fixed_point_align(mol, reference_mol, random_index, match, simi_mol_name, logger)
                if code_1 == 0:
                    mol_3d_addh.SetProp('_Name', f"{simi_mol_name}_index_{conf_idx}_matchIdx_{match_idx}")
                    mols.append(mol_3d_addh)
            else:
                AlignMol(mol, reference_mol, atomMap=match)

                ff = UFFGetMoleculeForceField(mol)
                for pi, ref_xi in cmap.items():
                    pIdx = ff.AddExtraPoint(ref_xi.x, ref_xi.y, ref_xi.z, fixed=True)
                    ff.AddDistanceConstraint(pIdx - 1, pi, 0, 0, 10.0)          #添加位置限制
                ff.Initialize()
                n = 10
                more = ff.Minimize(energyTol=1e-6, forceTol=1e-3)
                while more and n:
                    more = ff.Minimize(energyTol=1e-6, forceTol=1e-3)
                    n -= 1
                AlignMol(mol, reference_mol, atomMap=match)
                mol.SetProp('_Name', f"{simi_mol_name}_index_{conf_idx}_matchIdx_{match_idx}")
                mols.append(mol)
    total_sample_num = sample_num * len(candi_matches)
    return mols,total_sample_num


def candi_align_crystal(target_chembl_id, pdb_id, compound_id, target_dir, columns, logger):
    '''
    error_code:
    0: Successfully generated at least one conformation;
    1: "crystal_ligand is too large"(> 60);
    2: "candidate compound is too large"(> 60);
    3: "candidate compound larger than crystal ligand too much"(> 30);
    4: "crystal_ligand canno be read by rdkit";
    5: "mcs is None";
    6: aligned but 0 mols generated;
    '''
    error_code = 0
    input_file = str(target_dir/f"simi_0.7_fp_basic.csv")
    input_df = pd.read_csv(input_file, sep="\t",dtype={'cry_lig_name':str})   #有的PDB ID（如'4e93'）会被识别为数字,导致报错
    pdbid_compnd_df = input_df[(input_df['cry_lig_name']==pdb_id) & (input_df['similar_compounds_name']==compound_id)].copy()  #深复制；不会改变原有df
    candi_df = pd.DataFrame(columns=columns, index=list(range(len(pdbid_compnd_df))))
    row = list(pdbid_compnd_df.itertuples())[0]
    #initial candi_df
    candi_df['cry_lig_name'] = row.cry_lig_name
    candi_df['cry_lig_smiles'] = row.cry_lig_smiles
    candi_df['similar_compounds_name'] = row.similar_compounds_name
    candi_df['similarity'] = row.similarity
    candi_df['part_fix'] = 'No'

    remover = SaltRemover()
    simi_mol = remover.StripMol(Chem.MolFromSmiles(row.similar_compounds_smiles))
    simi_mol_name = row.similar_compounds_name
    reference_mol_from_smi = Chem.MolFromSmiles(row.cry_lig_smiles)
    candi_df['similar_compounds_smiles'] = Chem.MolToSmiles(simi_mol)
    candi_df['similar_compounds_an'] = simi_mol.GetNumAtoms()
    candi_df['cry_lig_an'] = reference_mol_from_smi.GetNumAtoms()

    if reference_mol_from_smi.GetNumAtoms() > 60:
        logger.warning(f"[{time.ctime()}] Error type 1: Heavy atom number of {pdb_id}_ligand > 60, skipped.")
        error_code = 1
        return error_code

    if simi_mol.GetNumAtoms() > 60:
        logger.warning(f"[{time.ctime()}] Error type 2: Heavy atom number of {simi_mol_name} > 60, skipped.")
        error_code = 2
        return error_code

    if simi_mol.GetNumAtoms() - reference_mol_from_smi.GetNumAtoms() > 30:
        logger.warning(f"[{time.ctime()}] Error type 3: {simi_mol_name} larger than crystal ligand too much, skipped.")
        error_code = 3
        return error_code

    #read crystal_ligand.mol2/sdf
    PDBbind_fixed_sdf_path = '/home/xli/dataset/PDBBind_v2019_general_fixed_sdf'
    cry_lig_fixed_sdf = f'{PDBbind_fixed_sdf_path}/{pdb_id}_ligand.fixed.sdf'
    pdbbind_dir = '/home/xli/dataset/PDBbind_v2019/general_structure_only'
    cry_lig_sdf = f'{pdbbind_dir}/{pdb_id}/{pdb_id}_ligand.sdf'
    cry_lig_mol2 = f'{pdbbind_dir}/{pdb_id}/{pdb_id}_ligand.mol2'
    if Path(cry_lig_fixed_sdf).exists() and Chem.SDMolSupplier(cry_lig_fixed_sdf)[0] is not None:
        reference_mol = Chem.SDMolSupplier(cry_lig_fixed_sdf)[0]
    elif Chem.SDMolSupplier(cry_lig_sdf)[0] is not None:
        reference_mol = Chem.SDMolSupplier(cry_lig_sdf)[0]
        logger.warning(f"[{time.ctime()}] There is no {pdb_id}_ligand.fixed.sdf " \
            f"OR it cannot be read by rdkit, use {pdb_id}_ligand.sdf.")
    elif Chem.MolFromMol2File(cry_lig_mol2) is not None:
        reference_mol = Chem.MolFromMol2File(cry_lig_mol2)
        logger.warning(f"[{time.ctime()}] {pdb_id}_ligand.sdf also cannot be read " \
            f"by rdkit, use {pdb_id}_ligand.mol2.")
    else:
        logger.warning(f"[{time.ctime()}] Error type 4: Both {pdb_id}_ligand(.fixed).sdf and " \
            f"{pdb_id}_ligand.mol2 cannot be read by rdkit! skipped.")
        error_code = 4
        return error_code

    #calculate mcs
    timeout = 10
    mcs = rdFMCS.FindMCS([reference_mol, simi_mol], timeout = timeout, completeRingsOnly=True)
    ref_mol = reference_mol
    if mcs.smartsString == '':
        mcs = rdFMCS.FindMCS([reference_mol_from_smi, simi_mol], timeout = timeout, completeRingsOnly=True)
        if mcs.smartsString != '':
            ref_mol = reference_mol_from_smi
            logger.warning(f"[{time.ctime()}] 'mcs' between {pdb_id}_ligand.mol2 " \
                f"and {simi_mol_name} is None, While 'mcs' between " \
                f"{pdb_id}_ligand.smi and {simi_mol_name} is not None, Use it.")
        else:
            logger.warning(f"[{time.ctime()}] Error type 5: 'mcs' between {pdb_id}_ligand.mol2/smi " \
                f"and {simi_mol_name} is BOTH None! skipped.")
            error_code = 5
            return error_code
    if mcs.numAtoms < 7:
        logger.warning(f"[{time.ctime()}] 'mcs.numAtoms' between {pdb_id}_ligand.mol2 " \
            f"and {simi_mol_name} < 7 when 'timeout' = {timeout}, extend to 3600.")
        timeout = 3600
        mcs = rdFMCS.FindMCS([ref_mol, simi_mol], timeout = timeout, completeRingsOnly=True)
    all_mcs = rdFMCS.FindMCS([ref_mol, simi_mol], timeout = timeout)
    if all_mcs.numAtoms - mcs.numAtoms > 6:
        logger.warning(f"[{time.ctime()}] ['all_mcs.numAtoms' - 'mcs.numAtoms'] between " \
            f"{pdb_id}_ligand.mol2 and {simi_mol_name} > 6, " \
            f"Recording it 'part_fix'!")
        candi_df['part_fix'] = 'Yes'
    
    diff_AN = max(reference_mol.GetNumAtoms(), simi_mol.GetNumAtoms()) - mcs.numAtoms   #计算原子数最**多**的小分子的原子数与MCS数的差异
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    candi_df['diff_an'] = diff_AN
    candi_df['mcs_smarts'] = mcs.smartsString
    candi_df['core_num'] = mcs.numAtoms

    # Constrained align
    mols,total_sample_num = constrained_align_sample_minimize(
        simi_mol=simi_mol, 
        reference_mol=reference_mol, 
        mcs_mol=mcs_mol, 
        simi_mol_name=simi_mol_name,
        logger=logger
        )

    candi_df['sample_num'] = len(mols)
    pdbid_compnd_dir = Path('.').resolve()
    if len(mols) != 0:
        aligned_sdf_path = str(pdbid_compnd_dir/f"{pdb_id}_{simi_mol_name}" \
            f"_aligned_sample_num_{len(mols)}_UFF.sdf")
        w1 = Chem.SDWriter(aligned_sdf_path)
        for m in mols:
            w1.write(m)
        w1.close()
        logger.info(f"[{time.ctime()}] After align, {target_chembl_id}_{pdb_id}_{simi_mol_name} " \
            f"generated {len(mols)} / {total_sample_num} conformation.")
        candi_df.to_csv(str(pdbid_compnd_dir/f"aligned_core_sample_num.csv"),
            sep = "\t", 
            index = False) 
    else:
        error_code = 6
        logger.warning(f"[{time.ctime()}] Error type 6: There is NO conformation generated for " \
            f"{target_chembl_id}_{pdb_id}_{simi_mol_name} after align.")
    return error_code


def align_by_rdkit(target_chembl_id, pdb_id, compound_id, target_dir):
    columns = ["cry_lig_name",
            "cry_lig_smiles",
            "cry_lig_an",
            "similar_compounds_name",
            "similar_compounds_smiles",
            "similar_compounds_an",
            "similarity",
            "core_num",
            "diff_an",
            "part_fix",
            "mcs_smarts",
            "sample_num",
            ]
    logger = logging.getLogger(__name__)
    error_code = candi_align_crystal(target_chembl_id, pdb_id, compound_id, target_dir, columns, logger)
    return error_code
