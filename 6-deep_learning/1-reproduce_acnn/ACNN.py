"""
Script that trains Atomic Conv models on PDBbind dataset.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import json
import shutil
import random
import argparse
import numpy as np
from rdkit import Chem
from pathlib import Path
from datetime import datetime as dt

import tensorflow as tf
import deepchem as dc
from acnn.pdbbind_datasets import load_pdbbind
from acnn.atomic_conv import AtomicConvModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-clust_file")
parser.add_argument("-max_epoch", type=int, default=100)
parser.add_argument("-patience", type=int, default=3)
parser.add_argument("-version", default='1')
parser.add_argument("-subset", default='PLIM_all')
parser.add_argument("-component", default='binding')
parser.add_argument("-split", default='random')
parser.add_argument("-seed", type=int, default=111)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-save_dir", default='/tmp')
parser.add_argument("-data_dir", default='/pubhome/xli02/project/PLIM/v2019_dataset')
parser.add_argument("-result_dir", default='result')
parser.add_argument("-reload", action='store_true')
parser.add_argument(
    "-shuffle", action='store_true', help='Shuffling or Curriculum Learning')
parser.add_argument("-shuf_labels", action='store_true')
parser.add_argument("-trans", action='store_true')
parser.add_argument("-feat_only", action='store_true')
parser.add_argument("-predict_atomic", action='store_true')
parser.add_argument("-timestamp", action='store_true')
parser.add_argument("-load_binding_pocket", action='store_true')
# parser.add_argument("-index_type", default='PLIM_all')
# parser.add_argument("-dataset_start_idx", type=int, default=0)
# parser.add_argument("-dataset_end_idx", type=int, default=162337)
args = parser.parse_args()

start = dt.now()

# np seed for split only
np.random.seed(args.seed)
# tf seed not work, every training will different.
tf.set_random_seed(args.seed)

frag1_num_atoms = 368  # for ligand atoms with Hs.
# frag2_num_atoms = 1350  # for pocket atoms with Hs
frag2_num_atoms = 24000  # for protein atoms

# frag2_num_atoms = 1067  # for pocket atoms without Hs
# frag2_num_atoms = 24000  # for protein atoms
complex_num_atoms = frag1_num_atoms + frag2_num_atoms

# args.split keep the name of splitting method.
split = args.split

# split based on clusters in json file
if args.clust_file:
  split = 'json'

# if using core set as test set, do not split when load_pdbbind.

pdbbind_tasks, pdbbind_datasets, transformers = load_pdbbind(
    reload=args.reload,
    featurizer="atomic",
    version=args.version,
    frag1_num_atoms=frag1_num_atoms,
    frag2_num_atoms=frag2_num_atoms,
    # shard_size=1024,
    split=split,
    split_seed=args.seed,
    clust_file=args.clust_file,
    shuf_labels=args.shuf_labels,
    subset=args.subset,
    load_binding_pocket=args.load_binding_pocket,
    # load_binding_pocket=False,
    data_dir=args.data_dir,
    save_dir=args.save_dir,
    save_timestamp=args.timestamp,
    transform=args.trans,
    # index_type=args.index_type,
    # dataset_start_idx=args.dataset_start_idx,
    # dataset_end_idx=args.dataset_end_idx
)


# constructing train, valid, and test sets.
# split is None when use PDBbind core set as test set.
if split is not None:
  train_dataset, valid_dataset, test_dataset = pdbbind_datasets


if args.feat_only:
  raise SystemExit(0)
# transformers = [
#     dc.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)
# ]
# for transformer in transformers:
#   train_dataset = transformer.transform(train_dataset)
#   valid_dataset = transformer.transform(valid_dataset)
#   test_dataset = transformer.transform(test_dataset)

metrics = [
    dc.metrics.Metric(dc.metrics.pearson_r2_score),
    dc.metrics.Metric(dc.metrics.mean_absolute_error)
]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# default
# atom_types=[6, 7, 8, 9, 11, 12, 15, 16, 17, 20, 25, 30, 35, 53, -1]
atom_types = [1, 6, 7, 8, 9, 12, 15, 16, 17, 20, 25, 30, 35, 53, -1]

# [[Rc],[Rs], [Re]], Rc is cutoff, Rs is mean, Re is variance.
default_radial = [[
    1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
    9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
], [0.0, 4.0, 8.0], [0.4]]

min_radial = [[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 2.0, 4.0], [0.4]]
model = AtomicConvModel(
    batch_size=args.batch_size,
    atom_types=atom_types,
    max_num_neighbors=4,
    radial=min_radial,
    frag1_num_atoms=frag1_num_atoms,
    frag2_num_atoms=frag2_num_atoms,
    complex_num_atoms=complex_num_atoms,
    component=args.component,
    learning_rate=0.001,
    configproto=config,
    model_dir=args.result_dir
)

# Fit trained model
print("Fitting model on train dataset")
patience = 0
best_r2 = 0
best_scores = None
train_evaluator = dc.utils.evaluate.Evaluator(model, train_dataset,
                                              transformers)
valid_evaluator = dc.utils.evaluate.Evaluator(model, valid_dataset,
                                              transformers)
test_evaluator = dc.utils.evaluate.Evaluator(model, test_dataset, transformers)


def copy_checkpoint(source, target='best_checkpoint'):
  import os
  from shutil import copyfile
  dirname = os.path.dirname(source)
  if '/' not in target:
    target = os.path.join(dirname, target)
  for item in os.listdir(dirname):
    item = os.path.join(dirname, item)
    name, ext = os.path.splitext(item)
    if name == source:
      copyfile(source + ext, target + ext)
  return target


if args.shuffle:
  # batches will be different in each epoch
  deterministic = False
  shuffle_batches = False
else:
  # only shuffle batches, samples in each batch will be same in each epoch.
  # samples sorted by pK in PDBbind data set.
  # each batch has samples with similar pK.
  # curriculum learning https://doi.org/10.1145/1553374.1553380
  deterministic = True
  shuffle_batches = True

for i in range(args.max_epoch):
  model.fit(
      train_dataset,
      nb_epoch=1,
      deterministic=deterministic,
      shuffle_batches=shuffle_batches)

  # print("Evaluating model at {} epoch".format(i + 1))
  print(f"[{time.ctime()}] Evaluating model at {i + 1} epoch")
  valid_scores = valid_evaluator.compute_model_performance(metrics)
  print("Validation scores")
  print(valid_scores, flush=True)
  if np.isnan(valid_scores['pearson_r2_score']):
    # break
    valid_scores['pearson_r2_score'] = 0
    valid_scores['mean_absolute_error'] = 0
    # continue
  if valid_scores['pearson_r2_score'] < best_r2:
    patience += 1
    if patience > args.patience:
      break
  else:
    last_checkpoint = model.get_checkpoints()[-1]
    best_checkpoint = copy_checkpoint(last_checkpoint)
    patience = 0
    best_r2 = valid_scores['pearson_r2_score']
    print('### Better on valid at epoch {}'.format(i + 1))
    test_scores = test_evaluator.compute_model_performance(metrics)
    print("Testing scores")
    print(test_scores)
    print(flush=True)

result_dir = Path(args.result_dir)
result_dir.mkdir(parents=True, exist_ok=True)
model.restore(checkpoint=best_checkpoint)
train_scores = train_evaluator.compute_model_performance(
    metrics, csv_out=result_dir/"train.csv")
valid_scores = valid_evaluator.compute_model_performance(
    metrics, csv_out=result_dir/"valid.csv")
test_scores = test_evaluator.compute_model_performance(
    metrics, csv_out=result_dir/"test.csv")

# model.save()

best_scores = {
    'train': train_scores,
    'valid': valid_scores,
    'test': test_scores
}
print('performance of model best on validation dataset:')
print(json.dumps(best_scores, indent=2))

with open(result_dir/'best_scores.json', 'w') as f:
  data = vars(args)
  data['best_scores'] = best_scores
  json.dump(data, f, indent=2)

with open(result_dir/'splitted_ids.json', 'w') as f:
  data['splitted_ids'] = {
      'train': list(train_dataset.ids),
      'valid': list(valid_dataset.ids),
      'test': list(test_dataset.ids)
  }
  json.dump(data, f, indent=2)

if args.predict_atomic:
  # write atomic contributions to binding affinity in occupancy in pdb file.
  atomic_dir = result_dir/'atomic'
  atomic_dir.mkdir(exist_ok=True)

  def set_occupancy(mol, occupancy):
    for atom, occ in zip(mol.GetAtoms(), occupancy):
      info = atom.GetPDBResidueInfo()
      info.SetOccupancy(float(occ))

  atomic = model.predict_atomic(test_dataset)
  lig_atomic, pro_atomic, com_atomic = atomic
  ids = test_dataset.ids
  mols = []
  for i, x in enumerate(test_dataset.X):
    lig_x, pro_x, com_x = x
    lig_m, lig_c, lig_n, lig_z = lig_x
    pro_m, pro_c, pro_n, pro_z = pro_x
    com_m, com_c, com_n, com_z = com_x
    if args.component == 'ligand':
      # com_atomic and lig_atomic are use same ligand as input.
      atomic_energies = com_atomic[i] - lig_atomic[i]
      atomic_energies = atomic_energies[lig_z != 0]
      m = lig_m
      name = f'atom.{ids[i]}_ligand.pdb'
    if args.component == 'protein':
      # com_atomic and pro_atomic are use same protein as input.
      atomic_energies = com_atomic[i] - pro_atomic[i]
      atomic_energies = atomic_energies[pro_z != 0]
      set_occupancy(pro_m, atomic_energies)
      m = pro_m
      name = f'atom.{ids[i]}_pocket.pdb'
    if args.component == 'binding':
      lig_energies = lig_atomic[i][lig_z != 0]
      pro_energies = pro_atomic[i][pro_z != 0]
      com_energies = com_atomic[i][com_z != 0]
      assert len(lig_energies) + len(pro_energies) == len(com_energies)
      atomic_energies = com_energies - np.hstack((lig_energies, pro_energies))
      m = com_m
      name = f'atom.{ids[i]}_complex.pdb'
    set_occupancy(m, atomic_energies)
    Chem.MolToPDBFile(m, str(atomic_dir/name))

if split is None:
  shutil.rmtree(train_dataset.data_dir)
  shutil.rmtree(valid_dataset.data_dir)

print(f"Elapsed time {dt.now()- start}")