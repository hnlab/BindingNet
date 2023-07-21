'''
Obtain "target_id" "pdb_id" "compound_id" from "all_target_pdbid_compound.list"
And run align, filter_clash, rescore, final_filter for EACH COMPOUND.
error_code:
    0: Successfully generated best conformation;
    1: "crystal_ligand is too large"(> 60);
    2: "candidate compound is too large"(> 60);
    3: "candidate compound larger than crystal ligand too much"(> 30);
    4: "crystal_ligand canno be read by rdkit";
    5: "mcs is None";
    6: aligned but 0 conformation generated;
    7: After clash filtering, obtain 0 conformation;
    8: Rescore Error
    9: After rescoring, 0 conformation left
    10: After filtering by cutoff of 'LIG_DELTA' and 'TOTAL_DELTA_FIX', 0 conformation left;
    11: After filtering by cutoff of 'core_RMSD', 0 conformation left;
'''
from pathlib import Path
from os import chdir
import subprocess
import argparse
import logging
import time

import align_by_rdkit
import filter_serious_clash
import filter_calculate_core_RMSD_extract_final_pose

parser = argparse.ArgumentParser(
    description="Run align, filter_clash, rescore, final_filter for EACH COMPOUND;",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '-t',
    '--target_chembl_id',
    default='CHEMBL301',
    help='CHEMBL id of target',
)
parser.add_argument(
    '-p',
    '--pdb_id',
    default='2fvd',
    help='pdb id',
)
parser.add_argument(
    '-c',
    '--compound_id',
    default='CHEMBL214183',
    help='compound id',
)
parser.add_argument(
    '-wd',
    '--work_dir',
    default='/home/xli/Documents/projects/ChEMBL-scaffold',
    help='work_dir',
)
parser.add_argument(
    '-ODN',
    '--output_dataset_name',
    default='v2019_dataset',
    help='output dataset name',
)
parser.add_argument(
    '-scc',
    '--serious_clash_cutoff',
    default=1,
    type=int,
    help='If min_distance between atoms of candi_lig and receptor < ' \
    'serious_clash_cutoff Angstrom(clash considered), ' \
    'remove this conformation for plop rescoring',
)
parser.add_argument(
    '-dlig',
    '--lig_delta',
    default=-20,
    type=int,
    help='If lig_delta <= lig_delta(twist or unreasonable considered), remove this conformation',
)
parser.add_argument(
    '-dtotal',
    '--total_delta_fix',
    default=100,
    type=int,
    help='If total_delta_fix > total_delta_fix(still clash considered), remove this conformation',
)
parser.add_argument(
    '-CR',
    '--coreRMSD',
    default=2,
    type=float,
    help='If coreRMSD > coreRMSD(not resemble crystal pose considered), remove this conformation',
)

args = parser.parse_args()
target_chembl_id = args.target_chembl_id
pdb_id = args.pdb_id
compound_id = args.compound_id
WRKDIR = Path(args.work_dir)
target_dir = WRKDIR/f'{args.output_dataset_name}/web_client_{target_chembl_id}'
serious_clash_cutoff = args.serious_clash_cutoff
lig_delta = args.lig_delta
total_delta_fix = args.total_delta_fix
coreRMSD = args.coreRMSD

def align_():
    logging.info(f"[{time.ctime()}] For {target_chembl_id}_{pdb_id}_{compound_id}:")
    logging.info(f"[{time.ctime()}] Starting align...")
    error_code_1 = align_by_rdkit.align_by_rdkit(
        target_chembl_id, pdb_id, compound_id, target_dir)
    if error_code_1 != 0:
        logging.warning(f"[{time.ctime()}] Exit after align, error_code = {error_code_1}.")
        exit()
    logging.info(f"[{time.ctime()}] Align done, at least one conformation generated.")

def filter_clash():
    logging.info(f"[{time.ctime()}] Starting filtering serious clash...")
    error_code_2 = filter_serious_clash.filter_serious_clash(
        target_chembl_id, pdb_id, compound_id, serious_clash_cutoff)
    if error_code_2 != 0:
        logging.warning(f"[{time.ctime()}] Exit after filtering serious clash, error_code = {error_code_2}.")
        exit()
    logging.info(f"[{time.ctime()}] Filter serious clash done, at least one no_clash conformation generated.")

def rescore():
    logging.info(f"[{time.ctime()}] Starting rescoring...")
    rescore_script = str(WRKDIR/"pipeline/pipeline_2/rescore.sh")
    rescore_sb = subprocess.run(
        f"bash {rescore_script} {target_chembl_id} {pdb_id} {compound_id} {str(WRKDIR)}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )
    logging.debug("\n" + rescore_sb.stdout.decode())
    logging.debug("\n" + rescore_sb.stderr.decode())
    logging.info(f"[{time.ctime()}] Rescore done, generated 'output.test.sorted' and 'top-cmxminlig.pdb'.")

def final_extract_pose():
    chdir(str(cwd))
    rescore_log = cwd / f"rescore.log"
    error_code_3 = 0
    with open(rescore_log, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "exit." in line:
            error_code_3 = 8
            break
    if error_code_3 == 0:
        logging.info(
            f"[{time.ctime()}] Starting calculating 'core_RMSD' and final filtering..."
        )
        error_code_4 = filter_calculate_core_RMSD_extract_final_pose.final(
            target_chembl_id, pdb_id, compound_id, target_dir, lig_delta, total_delta_fix, coreRMSD
        )
        if error_code_4 != 0:
            logging.warning(f"[{time.ctime()}] Exit after filtering by cutoff, error_code = {error_code_4}.")
            chdir(str(f"rescore/{target_chembl_id}_{pdb_id}_{compound_id}_pose"))
            subprocess.run(
                f"cp output.test.sorted top-cmxminlig.pdb ../*log ../../",
                shell=True,
            )
            chdir(str(cwd))
            subprocess.run(f"rm -r rescore conform_mindist_greater_1_ligand.sdf aligned_core_sample_num.csv", shell=True)
            exit()

        logging.info(f"[{time.ctime()}] Filtering by cutoff and extract final pose done, BEST conformation generated.")
        chdir(str(f"rescore"))
        subprocess.run(
            f"cp *log ../",
            shell=True,
        )
        chdir(str(cwd))
        subprocess.run(f"rm -r rescore conform_mindist_greater_1_ligand.sdf aligned_core_sample_num.csv", shell=True)
        logging.info(f"[{time.ctime()}] FINAL DONE.")
    else:
        logging.warning(
            f"[{time.ctime()}] Exit after rescore, error_code = {error_code_3}."
        )
        logging.info(f"[{time.ctime()}] Finally obtain 0 conformation.")
        # subprocess.run(
        #     f"tar -cf {compound_id}.tar .",
        #     shell=True,
        # )

cwd = Path('.').resolve()
log_file = "whole.log"
logging.basicConfig(filename=log_file, filemode="w", level=logging.DEBUG)
logging.info(f"[{time.ctime()}] current_dir: {str(cwd)}")

align_()
filter_clash()
rescore()
final_extract_pose()
