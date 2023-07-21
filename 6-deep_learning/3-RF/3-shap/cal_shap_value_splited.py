import pandas as pd
import numpy as np
import joblib
import shap
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="calculate shap value",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '-t',
    '--test_set',
    default='/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/train.csv',
    help='training set for calculating shap value',
)
parser.add_argument(
    '-ds',
    '--dataset_name',
    default='PDBbind_minimized_intersected_Uw',
    help='dataset name',
)
parser.add_argument(
    '-o',
    '--output_dir',
    default='/pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/shap_res',
    help='output dir',
)
parser.add_argument(
    '-idx',
    '--index',
    default='xaa',
    help='splited index',
)

args = parser.parse_args()
test_set = args.test_set
dataset_name = args.dataset_name
output_dir = args.output_dir
index = args.index

if not Path.exists(Path(f'{output_dir}/{dataset_name}')):
    Path.mkdir(Path(f'{output_dir}/{dataset_name}'), exist_ok=True)

features = pd.read_csv("/pubhome/xli02/project/PLIM/deep_learning/RFscore/featured_data/tow_datasets_features_6A.csv", sep='\t')
vina_title =['vina_gauss1',
            'vina_gauss2',
            'vina_repulsion',
            'vina_hydrophobic',
            'vina_hydrogen',
            'vina_num_rotors']
rf_v1_title = features.columns.tolist()[:36]
feature_list = vina_title+rf_v1_title

with open(f"/pubhome/xli02/project/PLIM/deep_learning/RFscore/test_res/{dataset_name}_Rm_core/1/best_model_VR1_RF.pkl", "rb") as f:
        model = joblib.load(f)
explainer = shap.TreeExplainer(model)

test_df = pd.read_csv(test_set, sep='\t', header=0, names=['unique_identify', '-logAffi'])
test_set = pd.merge(test_df, features, on=['unique_identify','-logAffi'])
print(f'There are {len(test_set)} for {dataset_name}_training_repeat1_{index}.')

shap_values = explainer.shap_interaction_values(test_set[feature_list])
np.save(f"{output_dir}/{dataset_name}/training_rep1_{index}.npy", shap_values)
