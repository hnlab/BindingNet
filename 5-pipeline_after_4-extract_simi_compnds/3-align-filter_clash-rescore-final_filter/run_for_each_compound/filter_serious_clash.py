"""
Calculat the distance among ligand atoms and protein atoms
 for each ligand conformation
Find min distance for each ligand atom
If min_distance < 1 Angstrom(clash considered),
 remove this conformation for plop minimizing
convert from sdf to mol2 for rescoring
"""
import time
import numpy as np
import logging
import subprocess
from scipy.spatial import distance
from pathlib import Path


def obtain_rec_coordi(rec_pdb):
    with open(rec_pdb, "r", newline="") as f1:
        lines_rec = f1.readlines()

    rec_heavy_atom_lines = []
    for line in lines_rec:
        if "ATOM" in line:
            if "           H" not in line:
                rec_heavy_atom_lines.append(line.rstrip("\n"))

    rec_heavy_atom_coordi = []
    for line in rec_heavy_atom_lines:
        rec_heavy_atom_coordi.append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    return rec_heavy_atom_coordi


def split_ligs_and_label_clash(ligs_sdf, rec_heavy_atom_coordi, serious_clash_cutoff):
    with open(ligs_sdf, "r", newline="") as f2:
        all_lines_ligs = f2.readlines()

    # split_ligs: retain all lines in sole sdf file
    ligs_with_h = []
    lig_with_h = []
    for line in all_lines_ligs:
        if "$$$$" in line:
            lig_with_h.append("$$$$")
            ligs_with_h.append(lig_with_h)
            lig_with_h = []
        else:
            lig_with_h.append(line.rstrip("\n"))

    index_clash = {}
    for lig_index in range(len(ligs_with_h)):
        lig_heavy_atom_coordi = []
        for line in ligs_with_h[lig_index]:
            if len(line) == 69 and "H   " not in line:
                lig_heavy_atom_coordi.append(
                    [float(line[0:10]), float(line[10:20]), float(line[20:30])]
                )

        # calculate distance and label clash
        distance_array = distance.cdist(
            np.array(lig_heavy_atom_coordi),
            np.array(rec_heavy_atom_coordi),
            "euclidean",
        )
        min_distance_for_each_lig_atom = np.amin(distance_array, axis=1)

        index_clash[lig_index] = False
        if (
            min(min_distance_for_each_lig_atom) < serious_clash_cutoff
        ):  # once the mini_distance of ligand atom < cutoff --> clash
            index_clash[lig_index] = True
    return ligs_with_h, index_clash


def write_no_clash_sdfs(ligs_with_h, index_clash, filtered_sdf_file):
    filtered_num = 0
    with open(filtered_sdf_file, "w", newline="") as f:
        for lig_index in range(len(ligs_with_h)):
            if index_clash[lig_index] == False:
                filtered_num = filtered_num + 1
                for line in ligs_with_h[lig_index]:
                    f.write(line + "\n")
    return filtered_num


def filter_serious_clash(target_chembl_id, pdb_id, compound_id, serious_clash_cutoff):
    logger = logging.getLogger(__name__)
    
    rec_pdb = f"/home/xli/dataset/PDBbind_v2019/general_structure_only/{pdb_id}/{pdb_id}_protein.pdb"
    rec_heavy_atom_coordi = obtain_rec_coordi(rec_pdb)

    pdbid_compnd_dir = Path('.').resolve()
    ligs_sdf = sorted(pdbid_compnd_dir.rglob("*aligned_sample*.sdf"))[0]
    ligs_with_h, index_clash = split_ligs_and_label_clash(
        ligs_sdf,
        rec_heavy_atom_coordi,
        serious_clash_cutoff=serious_clash_cutoff,
    )
    filtered_sdf_file = (
        pdbid_compnd_dir/f"conform_mindist_greater_{serious_clash_cutoff}_ligand.sdf"
    )
    filtered_num = write_no_clash_sdfs(
        ligs_with_h, index_clash, filtered_sdf_file
    )
    if filtered_num == 0:
        logging.warning(f"[{time.ctime()}] Error type 7: There is NO conformation generated for " \
            f"{target_chembl_id}_{pdb_id}_{compound_id} after clash filtering .")
        subprocess.run(f"rm conform_mindist_greater*.sdf", shell=True)
        error_code = 7
    else:
        error_code = 0
        logger.info(
            f"[{time.ctime()}] After clash filtering, {target_chembl_id}_{pdb_id}_{compound_id} "
            f"gets {filtered_num}/{len(ligs_with_h)} conformation."
        )
    return error_code
