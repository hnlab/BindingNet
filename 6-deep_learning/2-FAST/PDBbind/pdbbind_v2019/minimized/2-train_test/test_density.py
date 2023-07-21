################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network evaluation script
################################################################################


import os
import os.path as osp
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from tqdm import tqdm
from glob import glob
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import Data, Batch, DataListLoader
from data_utils import PDBBindDataset
from model import PotentialNetParallel
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def test(args):

    if torch.cuda.is_available():

        model_train_dict = torch.load(args.checkpoint)

    else:
        model_train_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    '''
    model = GeometricDataParallel(
        PotentialNetParallel(
            in_channels=20,
            out_channels=1,
            covalent_gather_width=model_train_dict["args"]["covalent_gather_width"],
            non_covalent_gather_width=model_train_dict["args"][
                "non_covalent_gather_width"
            ],
            covalent_k=model_train_dict["args"]["covalent_k"],
            non_covalent_k=model_train_dict["args"]["non_covalent_k"],
            covalent_neighbor_threshold=model_train_dict["args"]["covalent_threshold"],
            non_covalent_neighbor_threshold=model_train_dict["args"][
                "non_covalent_threshold"
            ],
        )
    ).float()
    
    '''
    model = PotentialNetParallel(
            in_channels=20,
            out_channels=1,
            covalent_gather_width=model_train_dict["args"]["covalent_gather_width"],
            non_covalent_gather_width=model_train_dict["args"][
                "non_covalent_gather_width"
            ],
            covalent_k=model_train_dict["args"]["covalent_k"],
            non_covalent_k=model_train_dict["args"]["non_covalent_k"],
            covalent_neighbor_threshold=model_train_dict["args"]["covalent_threshold"],
            non_covalent_neighbor_threshold=model_train_dict["args"][
                "non_covalent_threshold"
            ],
        ).float()
    
    model_module = torch.nn.Module()
    model_module.add_module('module', model)
    model = model_module

    print(model_module, model)
    

    model.load_state_dict(model_train_dict["model_state_dict"])

    dataset_list = []
    
    # because the script allows for multiple datasets, we iterate over the list of files to build one combined dataset object
    for data in args.test_data:
        dataset_list.append(
            PDBBindDataset(
                data_file=data,
                dataset_name=args.dataset_name,
                feature_type=args.feature_type,
                preprocessing_type=args.preprocessing_type,
                output_info=True,
                cache_data=False,
                use_docking=args.use_docking,
            )
        )

    dataset = ConcatDataset(dataset_list)
    print("{} complexes in dataset".format(len(dataset)))

    dataloader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    if torch.cuda.is_available():
    
        model.cuda()

    if args.print_model:
        print(model)
    print("{} total parameters.".format(sum(p.numel() for p in model.parameters())))

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    output_f = "{}/{}.hdf".format(args.output, args.subset_name) 

    names=[]
    with h5py.File(output_f, "w") as f:
        y_true = torch.tensor(())
        y_pred = torch.tensor(()).cuda()
        with torch.no_grad():
            for batch in tqdm(dataloader):

                batch = [x for x in batch if x is not None]
                if len(batch) < 1:
                    continue

                for item in batch:
                    name = item[0]
                    pose = item[1]
                    data = item[2]

                    names.append(name)
                    name_grp = f.require_group(str(name))

                    name_pose_grp = name_grp.require_group(str(pose))

                    y = data.y

                    name_pose_grp.attrs["y_true"] = y

                    (
                        covalent_feature,
                        non_covalent_feature,
                        pool_feature,
                        fc0_feature,
                        fc1_feature,
                        y_,
                    ) = model.module(
                        Batch().from_data_list([data]), return_hidden_feature=True
                    )
                    y_true = torch.cat((y_true, y), 0)
                    y_pred= torch.cat((y_pred, y_), 0)

                    name_pose_grp.attrs["y_pred"] = y_.cpu().data.numpy()
                    hidden_features = np.concatenate(
                        (
                            covalent_feature.cpu().data.numpy(),
                            non_covalent_feature.cpu().data.numpy(),
                            pool_feature.cpu().data.numpy(),
                            fc0_feature.cpu().data.numpy(),
                            fc1_feature.cpu().data.numpy(),
                        ),
                        axis=1,
                    )

                    name_pose_grp.create_dataset(
                        "hidden_features",
                        (hidden_features.shape[0], hidden_features.shape[1]),
                        data=hidden_features,
                    )
    y = y_true.cpu().data.numpy()
    y_ = y_pred.cpu().data.numpy()
    r2 = r2_score(y_true=y, y_pred=y_)
    mae = mean_absolute_error(y_true=y, y_pred=y_)
    mse = mean_squared_error(y_true=y, y_pred=y_)
    pearsonr = stats.pearsonr(y.reshape(-1), y_.reshape(-1))
    spearmanr = stats.spearmanr(y.reshape(-1), y_.reshape(-1))

    res_df = pd.DataFrame.from_dict({})
    res_df['unique_identify'] = names
    res_df['y_true'] = y
    res_df['y_pred'] = y_
    res_df.to_csv(f'{args.output}/{args.subset_name}.csv', sep='\t', index=False)

    #https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    # Calculate the point density
    # xy = np.vstack([y, y_])
    xy = np.vstack([y.T, y_.T])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y, y_, z = y[idx], y_[idx], z[idx]

    #https://github.com/hnlab/handbook/blob/41ad374cd0f9dc3ef882a7724eaac3d1f748fc05/0-General-computing-skills/MISC/vsfig.py#L83-L134
    fig, ax = plt.subplots()
    ax.scatter(y, y_, s=2, c=z, zorder=2)
    # ax.scatter(y, y_, c='red', zorder=2)
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f'{args.title}(N={len(y)})')
    ax.set_ylabel('predicted binding affinity')
    ax.set_xlabel('experimental binding affinity')
    # ax.scatter(y, y_)
    ax.text(0.5,0.99,
        f'R2:{r2:.3f}; MAE:{mae:.3f}; MSE:{mse:.3f};\npearsonr:{float(pearsonr[0]):.3f}; spearmanr:{float(spearmanr[0]):.3f}',
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
        zorder=3)
    fig.savefig(f'{args.output}/{args.subset_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Performance on {args.subset_name} set: r2: {r2}, mae: {mae}, mse: {mse}, pearsonr: {pearsonr}, spearmanr: {spearmanr}')

    # res_df = pd.DataFrame.from_dict({})
    # res_df['unique_identify'] = names
    # res_df['y_true'] = y
    # res_df['y_pred'] = y_
    # res_df.to_csv(f'{args.output}/{args.subset_name}.csv', sep='\t', index=False)
    
def main(args):
    test(args)


if __name__ == "__main__": 

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="path to model checkpoint")

    parser.add_argument(
        "--preprocessing-type",
        type=str,
        choices=["raw", "processed"],
        help="idicate raw pdb or (chimera) processed",
        required=True,
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["pybel", "rdkit"],
        help="indicate pybel (openbabel) or rdkit features",
        required=True,
    )
    parser.add_argument("--dataset-name", type=str, required=True)

    parser.add_argument(
        "--batch-size", type=int, default=1, help="batch size to use for dataloader"
    )

    parser.add_argument(
        "--num-workers", default=24, type=int, help="number of workers for dataloader"
    )

    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--output", help="path to output directory")
    parser.add_argument(
        "--use-docking",
        help="flag to indicate if dataset contains docking info",
        default=False,
        action="store_true",
    )
    parser.add_argument("--subset-name", help="subset name, including test, valid, train", default='test')
    parser.add_argument("--print-model", action="store_true", help="bool flag to determine whether to print the model")
    parser.add_argument("--title")
    args = parser.parse_args()


    main(args)
