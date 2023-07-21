import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_utils import PDBBindDataset
from torch.utils.data import ConcatDataset
from torch_geometric.data import DataListLoader

parser = argparse.ArgumentParser()
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
    "--batch_size", type=int, default=1, help="batch size to use for dataloader"
)
parser.add_argument("--test-data", nargs="+", required=True)
parser.add_argument("--output", help="path to output directory")
args = parser.parse_args()

dataset_list = []
for data in args.test_data:
    dataset_list.append(
        PDBBindDataset(
            data_file=data,
            dataset_name=args.dataset_name,
            feature_type=args.feature_type,
            preprocessing_type=args.preprocessing_type,
            output_info=True,
            cache_data=False,
            use_docking=False,
        )
    )

dataset = ConcatDataset(dataset_list)
print("{} complexes in dataset".format(len(dataset)))

dataloader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=False)

names = []
y_true = torch.tensor(())
with torch.no_grad():
    for batch in tqdm(dataloader):

        batch = [x for x in batch if x is not None]
        if len(batch) < 1:
            continue

        for item in batch:
            name = item[0]
            pose = item[1]
            data = item[2]

            y = data.y

            names.append(name)
            y_true = torch.cat((y_true, y), 0)

y = y_true.cpu().data.numpy()
res_df = pd.DataFrame.from_dict({'unique_identify': names, '-logAffi': [j for i in y.tolist() for j in i ]})
res_df.to_csv(args.output, sep = '\t', index=False)
