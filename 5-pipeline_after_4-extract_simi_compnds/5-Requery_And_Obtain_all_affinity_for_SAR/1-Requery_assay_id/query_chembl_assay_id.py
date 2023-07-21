"""
Scripts for query activities items from ChEMBL
corresponding targets from a list.
"""

from pathlib import Path
from threading import Thread, Lock, current_thread, active_count
from chembl_webresource_client.new_client import new_client
import csv
import logging
import time


def extract_targets(target_list):
  targets2chembl = {}
  header = True
  with open(target_list, 'r') as target_list:
    lines = target_list.readlines()
  for chembl_ids in [x.rstrip().split('\t')[2][:-1]     #必须要有"[:-1]"，去除最后的分号，才能用"chembl_ids.split(';')"
                       for x in lines]:
    if header:
      header = False
      continue
    for chembl_id in chembl_ids.split(';'):  #提取chembl_id，一行chembl_ids可能得到多个chembl_id；共1395个
      targets2chembl[chembl_id] = {}
  return targets2chembl

def prep_folders(out_dir, chembl_ids):
  if not Path.exists(out_dir):
    Path.mkdir(out_dir)

  query_files = [
      out_dir / f'tsv/web_client_{chembl_id}-activity.tsv'
      for chembl_id in chembl_ids
  ]
  smiles_files = [
      out_dir / f'smi/web_client_{chembl_id}-smiles-chembl_id.smi'
      for chembl_id in chembl_ids
  ]
  return query_files, smiles_files


def query_write_extract(idx):
  query_fields = [
  'activity_id',
  'assay_chembl_id',
  'assay_type',
  'target_chembl_id',
  'molecule_chembl_id',
  'canonical_smiles',
  'standard_type',
  'standard_relation',
  'standard_value',
  'standard_units',
  'pchembl_value',
  ]

  def query(idx):
    chembl_candi_lig_activities = new_client.activity.filter(
      target_chembl_id__in=chembl_ids[idx]).filter(
        standard_type__in=['IC50', 'Ki', 'EC50', 'Kd']).filter(
          standard_relation__in=['=', '<']).only(
            query_fields)
    return chembl_candi_lig_activities
  
  def write_list_of_dicts_to_csv(activities, query_file):
    with open(query_file, 'w', newline='') as query_file:
      writer = csv.DictWriter(query_file,
                              fieldnames=query_fields,
                              delimiter='\t',
                              quotechar='"',
                              quoting=csv.QUOTE_NONE)
      writer.writeheader()
      for activitie in activities:  #except query_fields, other related fields will be saved in activities
        del activitie['relation']
        del activitie['type']
        del activitie['units']
        del activitie['value']
        writer.writerow(activitie)

  def extract_smiles_chemblid(query_file, smiles_file):
    with open(query_file, 'r', newline='') as query_file:
      with open(smiles_file, 'w') as smiles_file:
        writer = csv.writer(smiles_file, delimiter='\t')
        for row in csv.reader(query_file, delimiter='\t'):
          # writer.writerow(row[3:1:-1])  # column smiles, molecule_chembbl_id
          writer.writerow(row[5:3:-1])  # column smiles, molecule_chembbl_id
    # TODO(mapleaf) diff activities have the same molecule_chembl_id
  
  if Path.is_file(smiles_files[idx]):
    logger.warning(f'[{time.ctime()}] SMIELS for target {chembl_ids[idx]:15s} exists, skipped.')
    with lock:
      remaining.remove(current_thread().name)
  else: 
    activities = query(idx)
    write_list_of_dicts_to_csv(activities, query_files[idx])
    extract_smiles_chemblid(query_files[idx], smiles_files[idx])
    with lock:
      logger.warning(
        f'[{time.ctime()}] Queried {len(activities):5d} results for '
        f'No.{idx+1:>5d} target {chembl_ids[idx]:15s};'
        f'{len(remaining)-1:5d} targets left; '
        f'Active conunts: {active_count()}'
        )
      remaining.remove(current_thread().name)


WRKDIR = Path('/home/xli/Documents/projects/ChEMBL-scaffold')

log_file = WRKDIR/f'pipeline/pipeline_2/10.0-Requery_assay_id/query.{time.strftime("%Y%m%d-%H%M%S")}.log'
logging.basicConfig(filename=log_file, filemode='w', level=logging.WARNING) 
logger = logging.getLogger(__name__)

targets2chembl = extract_targets(
  WRKDIR/f'pipeline/2-query_target_ChEMBLid/converted_PDBIDs_INDEX_general_PL_data.2019.tab.tsv')
chembl_ids = list(targets2chembl.keys())
logger.warning(
  f'[{time.ctime()}] Imported {len(chembl_ids)} chembl targets.')

out_dir = WRKDIR / 'pipeline/pipeline_2/10.0-Requery_assay_id/query_results'
query_files, smiles_files = prep_folders(out_dir, chembl_ids)

# chunk_size = 1
lock = Lock()

threads = [
  Thread(target=query_write_extract,
          args=(idx, )
        )
  for idx in range(0, len(chembl_ids))
]

# threads = [
#   Thread(target=query_write_extract,
#           args=(idx, )
#         )
#   for idx in range(211, 212)
# ]

remaining = [t.name for t in threads]
for t in threads:
  while active_count() > 50:    #不能太大，太大会被封
    time.sleep(1)
  t.start()   #开始运行各个线程

logger.warning(f'[{time.ctime()}] All started.')

while len(remaining) > 0:   #主线程等待，直至所有线程都运行完成？
  time.sleep(1)  # reduce much loops

