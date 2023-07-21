import os
from tfbio.data import Featurizer
import numpy as np
import h5py
import argparse
# import pybel
from openbabel import pybel
import warnings
#from data_generator.atomfeat_util import read_pdb, rdkit_atom_features, rdkit_atom_coords
#from data_generator.chem_info import g_atom_vdw_ligand, g_atom_vdw_protein
import xml.etree.ElementTree as ET
from rdkit.Chem.rdmolfiles import MolFromMol2File
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
# from pybel import Atom
import pandas as pd
from tqdm import tqdm

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

# TODO: compute rdkit features and store them in the output hdf5 file
# TODO: instead of making a file for each split, squash into one?

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
parser.add_argument("--metadata")
parser.add_argument("--dataset-name")
parser.add_argument("--rec", default="false_rec", choices=['false_rec', 'true_rec'])
parser.add_argument("--train_data")
parser.add_argument("--valid_data")
parser.add_argument("--test_data")
# parser.add_argument("--random_conform", action="store_true")

args = parser.parse_args()


def parse_element_description(desc_file):
    element_info_dict = {}
    element_info_xml = ET.parse(desc_file)
    for element in element_info_xml.getiterator():
        if "comment" in element.attrib.keys():
            continue
        else:
            element_info_dict[int(element.attrib["number"])] = element.attrib

    return element_info_dict


def parse_mol_vdw(mol, element_dict):
    vdw_list = []

    if isinstance(mol, pybel.Molecule):
        for atom in mol.atoms:
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.atomicnum) == 1:
                continue
            if int(atom.atomicnum) == 0:
                continue
            else:
                vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))

    elif isinstance(mol, rdkit.Chem.rdchem.Mol):
        for atom in mol.GetAtoms():
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.GetAtomicNum()) == 1:
                continue
            else:
                vdw_list.append(float(element_dict[atom.GetAtomicNum()]["vdWRadius"]))
    else:
        raise RuntimeError("must provide a pybel mol or an RDKIT mol")

    return np.asarray(vdw_list)


def featurize_pybel_complex(ligand_mol, pocket_mol, name, dataset_name):

    featurizer = Featurizer()
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge') 

    # get ligand features
    ligand_coords, ligand_features = featurizer.get_features(ligand_mol, molcode=1)

    if not (ligand_features[:, charge_idx] != 0).any():  # ensures that partial charge on all atoms is non-zero?
        raise RuntimeError("invalid charges for the ligand {} ({} set)".format(name, dataset_name))  

    # get processed pocket features
    pocket_coords, pocket_features = featurizer.get_features(pocket_mol, molcode=-1)
    if not (pocket_features[:, charge_idx] != 0).any():
        raise RuntimeError("invalid charges for the pocket {} ({} set)".format(name, dataset_name))   

    # center the coordinates on the ligand coordinates
    centroid_ligand = ligand_coords.mean(axis=0)
    ligand_coords -= centroid_ligand

    pocket_coords -= centroid_ligand
    data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)), 
                                np.concatenate((ligand_features, pocket_features))), axis=1) 

    return data

def featurize_pybel_lig(ligand_mol, name, dataset_name):

    featurizer = Featurizer()
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge') 

    # get ligand features
    ligand_coords, ligand_features = featurizer.get_features(ligand_mol, molcode=1)

    if not (ligand_features[:, charge_idx] != 0).any():  # ensures that partial charge on all atoms is non-zero?
        raise RuntimeError("invalid charges for the ligand {} ({} set)".format(name, dataset_name))  

    # center the coordinates on the ligand coordinates
    centroid_ligand = ligand_coords.mean(axis=0)
    ligand_coords -= centroid_ligand

    data = np.concatenate((ligand_coords, 
                                ligand_features), axis=1) 

    return data


def split_data(affinity_df, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    # random split into train/valid/test
    index = np.random.permutation(len(affinity_df))
    train_idx = index[:int(len(affinity_df)*train_ratio)]
    valid_idx = index[int(len(affinity_df)*train_ratio):int(len(affinity_df)*(train_ratio+valid_ratio))]
    test_idx = index[int(len(affinity_df)*(train_ratio+valid_ratio)):]
    # sort by affinity within subset
    test_df, valid_df, train_df = affinity_df.iloc[test_idx].sort_values(by=['-logAffi']), affinity_df.iloc[valid_idx].sort_values(by=['-logAffi']), affinity_df.iloc[train_idx].sort_values(by=['-logAffi'])
    test_df.to_csv(f'{args.output}/test.csv', sep='\t', index=False)
    valid_df.to_csv(f'{args.output}/valid.csv', sep='\t', index=False)
    train_df.to_csv(f'{args.output}/train.csv', sep='\t', index=False)
    return test_df, valid_df, train_df

def write_hdf(input_df, set_name, failure_dict, element_dict, process_type, dataset_name):
    print("\nfound {} complexes in {} dataset".format(len(input_df), set_name))
    with h5py.File(f'{args.output}/{set_name}_{args.rec}.hdf', 'w') as f:
        for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):
            name = row['pdb_id']
            affinity = row['-logAffi']
            candi_lig = f'{args.input}/{name}/cry_lig_opt_converted.sdf'
            rec_mol2 = f'{args.input}/{name}/rec_addcharge_pocket_6.mol2'
            
            grp = f.create_group(str(name))
            grp.attrs['affinity'] = affinity
            pybel_grp = grp.create_group("pybel")
            processed_grp = pybel_grp.create_group(process_type)
            
            try:
                crystal_ligand = next(pybel.readfile('sdf', f'{candi_lig}'))


            # do not add the hydrogens! they were already added in chimera and it would reset the charges
            except:
                error ="no ligand for {} ({} dataset)".format(name, set_name)
                warnings.warn(error)
                failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error) 
                continue

            # extract the van der waals radii for the ligand
            crystal_ligand_vdw = parse_mol_vdw(mol=crystal_ligand, element_dict=element_dict) 
            
            # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
            if len(crystal_ligand_vdw) < 1:
                error = "{} ligand consists purely of hydrogen, no heavy atoms to featurize".format(name)
                warnings.warn(error) 
                failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                continue

            if args.rec == 'false_rec':
                try:
                    crystal_data = featurize_pybel_lig(ligand_mol=crystal_ligand, name=name, dataset_name=set_name)
                except RuntimeError as error:
                    failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                    continue
                # enforce a constraint that the number of atoms for which we have features is equal to number for which we have VDW radii 
                assert crystal_ligand_vdw.shape[0] == crystal_data.shape[0]

                # END QUALITY CONTROL: made it past the try/except blocks....now featurize the data and store into the .hdf file 
                crystal_grp = processed_grp.create_group(dataset_name)
                crystal_grp.attrs["van_der_waals"] = crystal_ligand_vdw 
                crystal_dataset = crystal_grp.create_dataset("data", data=crystal_data, 
                                                    shape=crystal_data.shape, dtype='float32', compression='lzf') 
            elif args.rec == 'true_rec':
                try:
                    crystal_pocket = next(pybel.readfile('mol2', rec_mol2)) 

                except:
                    error = "no pocket for {} ({} dataset)".format(name, set_name)
                    warnings.warn(error)
                    failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                    continue


                # extract the van der waals radii for the pocket
                crystal_pocket_vdw = parse_mol_vdw(mol=crystal_pocket, element_dict=element_dict)
                # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
                if len(crystal_pocket_vdw) < 1:
                    error = "{} pocket consists purely of hydrogen, no heavy atoms to featurize".format(name)
                    warnings.warn(error) 
                    failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                    continue

                crystal_ligand_pocket_vdw = np.concatenate([crystal_ligand_vdw.reshape(-1), crystal_pocket_vdw.reshape(-1)], axis=0)
                try:
                    crystal_data = featurize_pybel_complex(ligand_mol=crystal_ligand, pocket_mol=crystal_pocket, name=name, dataset_name=set_name)
                except RuntimeError as error:
                    failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(set_name), failure_dict["error"].append(error)
                    continue
                
                # enforce a constraint that the number of atoms for which we have features is equal to number for which we have VDW radii 
                assert crystal_ligand_pocket_vdw.shape[0] == crystal_data.shape[0]

                # END QUALITY CONTROL: made it past the try/except blocks....now featurize the data and store into the .hdf file 
                crystal_grp = processed_grp.create_group(dataset_name)
                crystal_grp.attrs["van_der_waals"] = crystal_ligand_pocket_vdw 
                crystal_dataset = crystal_grp.create_dataset("data", data=crystal_data, 
                                                    shape=crystal_data.shape, dtype='float32', compression='lzf') 
    return failure_dict


def main():
    process_type = 'raw'
    print(f'dataset_name is {args.dataset_name}; process_type is {process_type}; receptor is {args.rec}')
    element_dict = parse_element_description("/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/1-featurize/elements.xml")
    failure_dict = {"name": [], "partition": [], "set": [], "error": []}

    if not os.path.exists(args.output):
        os.makedirs(args.output) 

    if args.metadata:
        affinity_data = pd.read_csv(args.metadata, sep = "\t")
        print(f'There are {len(affinity_data)} compounds in {args.dataset_name}, and will be randomly splitted into train/valid/test set(8/1/1).')
        test_df, valid_df, train_df = split_data(affinity_data)
        train_name = 'PDBbind_v19_minimized_train'
        failure_dict = write_hdf(train_df, train_name, failure_dict, element_dict, process_type, args.dataset_name)
        valid_name = 'PDBbind_v19_minimized_valid'
        failure_dict = write_hdf(valid_df, valid_name, failure_dict, element_dict, process_type, args.dataset_name)
        test_name = 'PDBbind_v19_minimized_test'
        failure_dict = write_hdf(test_df, test_name, failure_dict, element_dict, process_type, args.dataset_name)
    if args.train_data:
        train_df_no_sort = pd.read_csv(args.train_data, sep = "\t")
        train_df = train_df_no_sort.sort_values(by=['-logAffi'])
        train_name = 'PDBbind_v19_minimized_train'
        print(f'There are {len(train_df)} compounds in {train_name}.')
        failure_dict = write_hdf(train_df, train_name, failure_dict, element_dict, process_type, args.dataset_name)
    if args.valid_data:
        valid_df_no_sort = pd.read_csv(args.valid_data, sep = "\t")
        valid_df = valid_df_no_sort.sort_values(by=['-logAffi'])
        valid_name = 'PDBbind_v19_minimized_valid'
        print(f'There are {len(valid_df)} compounds in {valid_name}.')
        failure_dict = write_hdf(valid_df, valid_name, failure_dict, element_dict, process_type, args.dataset_name)
    if args.test_data:
        test_df_no_sort = pd.read_csv(args.test_data, sep = "\t")
        test_df = test_df_no_sort.sort_values(by=['-logAffi'])
        test_name = 'PDBbind_v19_minimized_test'
        print(f'There are {len(test_df)} compounds in {test_name}.')
        failure_dict = write_hdf(test_df, test_name, failure_dict, element_dict, process_type, args.dataset_name)

    failure_df = pd.DataFrame(failure_dict)
    failure_df.to_csv("{}/failure_summary.csv".format(args.output), index=False)

if __name__ == "__main__":
    main()