"""
PDBBind dataset loader.
"""

from __future__ import division
from __future__ import unicode_literals

import logging
import multiprocessing
import os
import re
import time

import deepchem
import numpy as np
import pandas as pd
import tarfile

from deepchem.splits import Splitter
from deepchem.utils import rdkit_util, pad_array
from deepchem.utils.rdkit_util import MoleculeLoadException
from deepchem.feat import ComplexFeaturizer
from deepchem.feat.atomic_coordinates import compute_neighbor_list
from deepchem.feat.atomic_coordinates import NeighborListComplexAtomicCoordinates

logger = logging.getLogger(__name__)

class SimpleComplexNeighborListFragmentAtomicCoordinates(ComplexFeaturizer):
  """This class computes the featurization that corresponds to AtomicConvModel.

  This class computes featurizations needed for AtomicConvModel. Given a
  two molecular structures, it computes a number of useful geometric
  features. In particular, for each molecule and the global complex, it
  computes a coordinates matrix of size (N_atoms, 3) where N_atoms is the
  number of atoms. It also computes a neighbor-list, a dictionary with
  N_atoms elements where neighbor-list[i] is a list of the atoms the i-th
  atom has as neighbors. In addition, it computes a z-matrix for the
  molecule which is an array of shape (N_atoms,) that contains the atomic
  number of that atom.

  Since the featurization computes these three quantities for each of the
  two molecules and the complex, a total of 9 quantities are returned for
  each complex. Note that for efficiency, fragments of the molecules can be
  provided rather than the full molecules themselves.
  """

  def __init__(self,
               frag1_num_atoms,
               frag2_num_atoms,
               complex_num_atoms,
               max_num_neighbors,
               neighbor_cutoff,
               strip_hydrogens=True):
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.complex_num_atoms = complex_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.neighbor_cutoff = neighbor_cutoff
    self.strip_hydrogens = strip_hydrogens
    self.neighborlist_featurizer = NeighborListComplexAtomicCoordinates(
        self.max_num_neighbors, self.neighbor_cutoff)

  def _featurize_complex(self, mol_pdb_file, protein_pdb_file):
    try:
      frag1_coords, frag1_mol = rdkit_util.load_molecule(
          mol_pdb_file,
          add_hydrogens=False,
          calc_charges=False,
          sanitize=False
      )
      frag2_coords, frag2_mol = rdkit_util.load_molecule(
          protein_pdb_file,
          add_hydrogens=False,
          calc_charges=False,
          sanitize=False
      )
    # except MoleculeLoadException:
    except:
      # Currently handles loading failures by returning None
      # TODO: Is there a better handling procedure?
      logging.warning("Some molecules cannot be loaded by Rdkit. Skipping")
      return None
    system_mol = rdkit_util.merge_molecules(
        frag1_mol, frag2_mol)
    system_coords = rdkit_util.get_xyz_from_mol(system_mol)
    try:
      frag1_coords, frag1_neighbor_list, frag1_z = self.featurize_mol(
          frag1_coords, frag1_mol, self.frag1_num_atoms)

      frag2_coords, frag2_neighbor_list, frag2_z = self.featurize_mol(
          frag2_coords, frag2_mol, self.frag2_num_atoms)

      system_coords, system_neighbor_list, system_z = self.featurize_mol(
          system_coords, system_mol, self.complex_num_atoms)
    except ValueError as e:
      logging.warning(e)
      logging.warning(
          "max_atoms was set too low. Some complexes too large and skipped")
      return None

    return (frag1_mol, frag1_coords, frag1_neighbor_list, frag1_z), (frag2_mol, frag2_coords, frag2_neighbor_list, frag2_z), \
           (system_mol, system_coords, system_neighbor_list, system_z)

  def get_Z_matrix(self, mol, max_atoms):
    if len(mol.GetAtoms()) > max_atoms:
      raise ValueError(f"A molecule (#atoms = {len(mol.GetAtoms())}) is larger than permitted by max_atoms. "
                       "Increase max_atoms and try again.")
    return pad_array(
        np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]), max_atoms)

  def featurize_mol(self, coords, mol, max_num_atoms):
    logging.info("Featurizing molecule of size: %d", len(mol.GetAtoms()))
    neighbor_list = compute_neighbor_list(coords, self.neighbor_cutoff,
                                          self.max_num_neighbors, None)
    z = self.get_Z_matrix(mol, max_num_atoms)
    z = pad_array(z, max_num_atoms)
    coords = pad_array(coords, (max_num_atoms, 3))
    return coords, neighbor_list, z

class JSONSplitter(Splitter):
  """
    Class for doing data splits based on clusters in JSON file.
    It split big/medium/small clusters into train/valid/test subsets.
  """

  def __init__(self, clust_file, *args, **kwargs):
    self.clust_file = clust_file
    self.ids_weight = {}
    super(JSONSplitter, self).__init__(*args, **kwargs)

  def train_valid_test_split(self,
                             dataset,
                             train_dir=None,
                             valid_dir=None,
                             test_dir=None,
                             frac_train=.8,
                             frac_valid=.1,
                             frac_test=.1,
                             seed=None,
                             log_every_n=1000,
                             verbose=True,
                             **kwargs):
    """
      Splits self into train/validation/test sets.

      Returns Dataset objects.
      """
    train_inds, valid_inds, test_inds = self.split(
        dataset,
        seed=seed,
        frac_train=frac_train,
        frac_test=frac_test,
        frac_valid=frac_valid,
        log_every_n=log_every_n,
        **kwargs)
    import tempfile
    if train_dir is None:
      train_dir = tempfile.mkdtemp()
    if valid_dir is None:
      valid_dir = tempfile.mkdtemp()
    if test_dir is None:
      test_dir = tempfile.mkdtemp()

    train_dataset = dataset.select(train_inds, train_dir)
    if frac_valid != 0:
      valid_dataset = dataset.select(valid_inds, valid_dir)
    else:
      valid_dataset = None
    test_dataset = dataset.select(test_inds, test_dir)

    return train_dataset, valid_dataset, test_dataset

  def split(self,
            dataset,
            seed=None,
            frac_train=.8,
            frac_valid=.1,
            frac_test=.1,
            log_every_n=1000):
    """
      Splits proteins into train/validation/test by sequence clustering.
    """
    with open(self.clust_file) as f:
      import json
      clust_ids = json.load(f)
    for clust in clust_ids:
      for id_ in clust:
        self.ids_weight[id_] = 1.0 / len(clust)

    dataset_ids = dataset.ids
    weights = np.ones_like(dataset_ids)
    for i, id_ in enumerate(dataset_ids):
      if id_ in self.ids_weight:
        weights[i] = self.ids_weight[id_]
      else:
        self.ids_weight[id_] = 1.0

    # shuffle for not stable sort
    np.random.seed(seed)
    shuff_ids = np.random.permutation(len(weights))
    shuff_ws = weights[shuff_ids]
    
    # split big/medium/small clusters into train/valid/test subsets.
    # sort index by weight (1/clust_size).
    inds = shuff_ids[np.argsort(shuff_ws)]

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.)
    data_len = len(dataset)
    train_cutoff = int(frac_train * data_len)
    valid_cutoff = int((frac_train + frac_valid) * data_len)
    train_inds = inds[:train_cutoff]
    valid_inds = inds[train_cutoff:valid_cutoff]
    test_inds = inds[valid_cutoff:]
    return train_inds, valid_inds, test_inds


def load_pdbbind(reload=True,
                 data_dir=None,
                 version="2015",
                 subset="core",
                 frag1_num_atoms=350,
                 frag2_num_atoms=1350,
                 shard_size=4096,
                 load_binding_pocket=True,
                 featurizer="atomic",
                 split="random",
                 split_seed=None,
                 clust_file=None,
                 shuf_labels=False,
                 reweight=True,
                 save_dir=None,
                 transform=False,
                 save_timestamp=False):
  """Load raw PDBBind dataset by featurization and split.

  Parameters
  ----------
  reload: Bool, optional
    Reload saved featurized and splitted dataset or not.
  data_dir: Str, optional
    Specifies the directory storing the raw dataset.
  load_binding_pocket: Bool, optional
    Load binding pocket or full protein.
  subset: Str
    Specifies which subset of PDBBind, only "core" or "refined" for now.
  shard_size: Int, optinal
    Specifies size of shards when load general_PL subset considering its large scale.
    Default values are None for core/refined subset and 4096 for general_PL subset.
  featurizer: Str
    Either "grid" or "atomic" for grid and atomic featurizations.
  split: Str
    Either one of "random", "index", "fp", "mfp" and "seq" for random, index, ligand
    Fingerprints, butina clustering with Morgan Fingerprints of ligands, sequence 
    clustering of proteins splitting.
  split_seed: Int, optional
    Specifies the random seed for splitter.
  save_dir: Str, optional
    Specifies the directory to store the featurized and splitted dataset when
    reload is False. If reload is True, it will load saved dataset inside save_dir. 
  save_timestamp: Bool, optional
    Save featurized and splitted dataset with timestamp or not. Set it as True
    when running similar or same jobs simultaneously on multiple compute nodes.
  """

  pdbbind_tasks = ["-logKd/Ki"]  #标签列的列名组成

  deepchem_dir = deepchem.utils.get_data_dir()

  if data_dir == None:
    data_dir = deepchem_dir
  data_folder = data_dir


  if save_dir == None:
    save_dir = deepchem_dir
  if load_binding_pocket:
    feat_dir = os.path.join(save_dir, "feat-PLIM", "v" + version,
                            "protein_pocket-%s-%s" % (subset, featurizer))
  else:
    feat_dir = os.path.join(save_dir, "feat-PLIM", "v" + version,
                            "full_protein-%s-%s" % (subset, featurizer))

  if save_timestamp:
    feat_dir = "%s-%s-%s" % (feat_dir, time.strftime("%Y%m%d",
                                                     time.localtime()),
                             re.search(r"\.(.*)", str(time.time())).group(1))

  loaded = False
  if split is not None:
    if split_seed:
      split_dir = os.path.join(feat_dir, split + str(split_seed))
    else:
      split_dir = os.path.join(feat_dir, str(split))
    if transform:
      split_dir += '.trans'
    if reload:
      print("\nReloading splitted dataset from:\n%s\n" % split_dir)
      loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
          split_dir)
      if loaded:
        return pdbbind_tasks, all_dataset, transformers
      else:
        print('Fail to reload splitted dataset.')

  if reload and loaded == False:
    print("Reloading featurized dataset:\n%s\n" % feat_dir)
    try:
      dataset = deepchem.data.DiskDataset(feat_dir)
      loaded = True
    except ValueError:
      print('Fail to reload featurized dataset.')

  if loaded == False:
    print('Start to featurize dataset form raw data ...')
    if os.path.exists(data_folder):
      logger.info("PDBBind full dataset already exists.")
    else:
      dataset_file = os.path.join(data_dir, "pdbbind_v2015.tar.gz")
      if not os.path.exists(dataset_file):
        logger.warning(
            "About to download PDBBind full dataset. Large file, 2GB")
        deepchem.utils.download_url(
            'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/' +
            "pdbbind_v2015.tar.gz",
            dest_dir=data_dir)

      print("Untarring full dataset...")
      deepchem.utils.untargz_file(
          dataset_file, dest_dir=os.path.join(data_dir, "pdbbind"))

    print("\nRaw dataset:\n%s" % data_folder)
    print("\nFeaturized dataset:\n%s" % feat_dir)


    index_labels_file = os.path.join(data_folder, "index", "PLIM_dataset_v1_final.csv")

    # Extract locations of data
    index_df = pd.read_csv(index_labels_file, sep = "\t")
    labels = np.array(list(index_df['-logAffi']))
    unique_indentify = list(index_df['unique_indentify'])
    ligand_files = []
    protein_files = []
    for uniq_ind in unique_indentify:
        target = uniq_ind.split('_')[0]
        pdbid = uniq_ind.split('_')[1]
        compnd = uniq_ind.split('_')[2]
        target_pdb_dir = f'{data_folder}/web_client_{target}/{target}_{pdbid}'
        ligand_files.append(f'{target_pdb_dir}/{compnd}/{uniq_ind}_dlig_-20_dtotal_100_CoreRMSD_2.0_final.pdb')
        if load_binding_pocket:
            protein_files.append(f'{target_pdb_dir}/{compnd}/{uniq_ind}_pocket.pdb')
        else:
            protein_files.append(f'/home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/false_rec.pdb')


    # Featurize Data

    if featurizer == "atomic":
      # Pulled from PDB files. For larger datasets with more PDBs, would use
      # max num atoms instead of exact.
      complex_num_atoms = frag1_num_atoms + frag2_num_atoms
      max_num_neighbors = 4
      # Cutoff in Angstrom? but mdtraj use nm as default
      neighbor_cutoff = 4

      featurizer = SimpleComplexNeighborListFragmentAtomicCoordinates(
            frag1_num_atoms=frag1_num_atoms,
            frag2_num_atoms=frag2_num_atoms,
            complex_num_atoms=complex_num_atoms,
            max_num_neighbors=max_num_neighbors,
            neighbor_cutoff=neighbor_cutoff)

    def get_shards(inputs, shard_size):   #作为生成器不断返回 子inputs
      if len(inputs) <= shard_size:
        yield inputs
      else:
        assert isinstance(shard_size, int) and 0 < shard_size <= len(inputs)
        print("About to start loading files.\n")
        for shard_ind in range(len(inputs) // shard_size + 1):
          if (shard_ind + 1) * shard_size < len(inputs):
            print("Loading shard %d of size %s." % (shard_ind + 1,
                                                    str(shard_size)))
            yield inputs[shard_ind * shard_size:(shard_ind + 1) * shard_size]
        else:
          print("\nLoading shard %d of size %s." %
                (shard_ind + 1, str(len(inputs) % shard_size)))
          yield inputs[shard_ind * shard_size:len(inputs)]

    def shard_generator(inputs, shard_size):
      for shard_num, shard in enumerate(get_shards(inputs, shard_size)):
        time1 = time.time()
        ligand_files, protein_files, labels, pdbs = zip(*shard)
        features, failures = featurizer.featurize_complexes(    #提取特征 -> (成功的特征量, 提取特征量失败的index)
            ligand_files, protein_files)
        labels = np.delete(labels, failures)  #去除features中计算失败的特征
        labels = labels.reshape((len(labels), 1))
        weight = np.ones_like(labels)
        ids = np.delete(pdbs, failures)  #去除features中计算失败的特征;  ids: 每个样本的唯一标识符
        assert len(features) == len(labels) == len(weight) == len(ids)
        time2 = time.time()
        print("[%s] Featurizing shard %d took %0.3f s\n" % (time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()), shard_num, time2 - time1))
        yield features, labels, weight, ids

    print(
        "\n[%s] Featurizing and constructing dataset without failing featurization for"
        % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "\"%s\"\n" % data_folder)
    feat_t1 = time.time()
    if shuf_labels:
      labels = np.random.permutation(labels)
    sort_inds = np.argsort(labels)
    ligand_files = np.array(ligand_files)[sort_inds]
    protein_files = np.array(protein_files)[sort_inds]
    labels = labels[sort_inds]
    unique_indentify = np.array(unique_indentify)[sort_inds]
    zipped = list(zip(ligand_files, protein_files, labels, unique_indentify))
    dataset = deepchem.data.DiskDataset.create_dataset(
        shard_generator(zipped, shard_size),
        data_dir=feat_dir,
        tasks=pdbbind_tasks,
        verbose=True)

    print(f"Succeeded to featurize {len(dataset)}/{len(ligand_files)} samples.")
    feat_t2 = time.time()
    print("\n[%s] Featurization and construction finished, %0.3f s passed.\n" %
          (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
           feat_t2 - feat_t1))

  # Default: No transformations of data
  if transform:
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=dataset)
    ]
  else:
    transformers = []
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  # Split dataset
  print("\nSplit dataset...\n")
  if split == None:
    return pdbbind_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'json':JSONSplitter(clust_file),
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset, seed=split_seed)

  all_dataset = (train, valid, test)
  print("\nSaving dataset to \"%s\" ..." % split_dir)
  deepchem.utils.save.save_dataset_to_disk(split_dir, train, valid, test,
                                           transformers)
  return pdbbind_tasks, all_dataset, transformers