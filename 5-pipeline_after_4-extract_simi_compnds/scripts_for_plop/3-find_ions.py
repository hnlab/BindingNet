"""
Deal with the receptor containing ions:
find ions close to(< 3A) the crystal ligand;
modify the protonation state of residues close to(< 3A) ions:
    HIS -> HID/HIE
    CYS -> CYM
    LYS -> LYN
    ARG -> ARN
    TYR -> TYM
output "rec_modfied.pdb":
    "ATOM" with modified residues
    add "TER"
    add ions
        Change `occupancy` of ions to "1.00" -- ions can be recognized by PLOP
        `chainID` != ` ` and `resSeq` != `  1`
        `chainID_resSeq` CANNOT be the same for same ions
"""
import sys
import numpy as np
from scipy.spatial import distance

rec_pdb = sys.argv[1]
cry_lig_pdb = sys.argv[2]
rec_modified_pdb = sys.argv[3]


def obtain_lig_coordi(cry_lig_pdb):
    with open(cry_lig_pdb, "r", newline="") as f1:
        all_lines_ligs = f1.readlines()

    lig_heavy_atom_coordi = []
    for line in all_lines_ligs:
        if "           H" not in line:
            lig_heavy_atom_coordi.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return lig_heavy_atom_coordi


def obtain_ions_close_to_crylig(rec_pdb, lig_heavy_atom_coordi):
    with open(rec_pdb, "r", newline="") as f2:
        lines_rec = f2.readlines()

    close_lig_ions_lines = {}
    close_lig_ions_coordi = {}
    rec_atoms_lines = []
    rec_atoms_coordi = []
    for line in lines_rec:
        if "COMPND" in line:
            pdbid = line[10:14]
        if "ATOM" in line:
            rec_atoms_lines.append(line)
            rec_atoms_coordi.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
        if "HETATM" in line:
            if line[76:78] != "":
                element = line[76:78]
            else:
                element = line[17:20].split()[0]
            if element in ions:
                ions_seq = line[17:26]
                ions_coordi = [
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                ]
                distance_array = distance.cdist(
                    np.array(ions_coordi), 
                    np.array(lig_heavy_atom_coordi), 
                    "euclidean"
                )
                if np.amin(distance_array, axis=1) < 3:
                    close_lig_ions_lines[ions_seq] = line
                    close_lig_ions_coordi[ions_seq] = [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                    print(
                        f"Receptor of {pdbid} has ions {ions_seq}, " \
                            f"and it is close to crystal ligand (< 3A).\n"
                        f"PLEASE CHECK the protonation state of ligands " \
                            f"AFTER ALL THINGS DONE!"
                    )
                else:
                    print(
                        f"Receptor of {pdbid} has ions {ions_seq}, " \
                            f"but it is far away from crystal ligand (>= 3A).\n"
                        f"Ignore it temporarily, But PLEASE ADD IT AND " \
                            f"CHECK AFTER ALL THINGS DONE!"
                    )
    if len(close_lig_ions_coordi) != 0:
        dist_array = distance.cdist(
            np.array(list(close_lig_ions_coordi.values())),
            np.array(rec_atoms_coordi),
            "euclidean",
        )
        close_res_idx_array = np.where(dist_array < 3)[1]
        print(
            f"Receptor of {pdbid} has {len(close_lig_ions_coordi)} ions " \
                f"to be loaded in PLOP."
        )
        return rec_atoms_lines, close_res_idx_array, close_lig_ions_lines
    else:
        print(f"Receptor of {pdbid} has NO ions to be loaded in PLOP.")
        return rec_atoms_lines, [], close_lig_ions_lines


def change_res_protonation(close_res_idx_array, rec_atoms_lines):
    rec_seq_to_res = {}
    for idx in close_res_idx_array:
        line = rec_atoms_lines[idx]
        if line[17:20] in res_list:
            atom_name = line[12:16]
            res_name = line[17:20]
            res_chain_seq = line[21:26]
            if res_name == "HIS":
                if "NE2" in atom_name:
                    rec_seq_to_res[res_chain_seq] = "HID"
                elif "ND1" in atom_name:
                    rec_seq_to_res[res_chain_seq] = "HIE"
            elif res_name == "CYS" and "SG" in atom_name:
                rec_seq_to_res[res_chain_seq] = "CYM"
            elif res_name == "LYS" and "NZ" in atom_name:
                rec_seq_to_res[res_chain_seq] = "LYN"
            elif res_name == "ARG" and atom_name.split()[0] in ["NH1", "NH2", "NE"]:
                rec_seq_to_res[res_chain_seq] = "ARN"
            elif res_name == "TYR" and "OH" in atom_name:
                rec_seq_to_res[res_chain_seq] = "TYM"

    rec_modified_lines = []
    for line in rec_atoms_lines:
        if line[21:26] in rec_seq_to_res.keys():
            res_chain_seq = line[21:26]
            newline = line[:17] + rec_seq_to_res[res_chain_seq] + line[20:]
            rec_modified_lines.append(newline)
        else:
            rec_modified_lines.append(line)
    return rec_modified_lines


ions = [
    "AL",
    "CA",
    "CD",
    "CL",
    "CO",
    "CU",
    "EU",
    "F",
    "FE",
    "GD",
    "HG",
    "K",
    "LA",
    "LI",
    "MG",
    "MN",
    "MO",
    "NA",
    "NI",
    "OS",
    "PB",
    "RE",
    "SC",
    "SR",
    "YB",
    "ZN",
    "IN",
    "GA",
    "AU",
    "CS",
]
# temp_mine_addition: 'CM', 'CN', 'MC', 'NH', 'NO', 'NP', 'U'
res_list = ["HIS", "CYS", "LYS", "ARG", "TYR"]

lig_heavy_atom_coordi = obtain_lig_coordi(cry_lig_pdb)
(
    rec_atoms_lines,
    close_res_idx_array,
    close_lig_ions_lines,
) = obtain_ions_close_to_crylig(rec_pdb, lig_heavy_atom_coordi)
rec_modified_lines = change_res_protonation(close_res_idx_array, rec_atoms_lines)
with open(rec_modified_pdb, "w", newline="") as output_f:
    for line in rec_modified_lines:
        output_f.write(line)
    output_f.write("TER\n")
    for ions_idx in range(len(close_lig_ions_lines)):
        line = list(close_lig_ions_lines.values())[ions_idx]
        newline = (
            line[:21] + "8" + f"{ions_idx+1:>4}" + line[26:54] + "  1.00" + line[60:]
        )  # `occupancy` must be '  1.00' and `chainID_resSeq` cannot be '   1' and `chainID_resSeq` cannot be same for same ions
        print(f'Change the chainID_resSeq of {list(close_lig_ions_lines.keys())[ions_idx]} ' \
            f'to "8{ions_idx+1:>4}" in "rec_modified_pdb".')
        output_f.write(newline)
