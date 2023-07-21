import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

shap_dir = '/pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/split_shap_res/rm_core_ids'
features = pd.read_csv("/pubhome/xli02/project/PLIM/deep_learning/RFscore/featured_data/tow_datasets_features_6A.csv", sep='\t')
vina_title =['vina_gauss1',
            'vina_gauss2',
            'vina_repulsion',
            'vina_hydrophobic',
            'vina_hydrogen',
            'vina_num_rotors']
rf_v1_title = features.columns.tolist()[:36]
feature_list = vina_title+rf_v1_title

# change feature names
## 6.7(l.p) --> N-C(p.l)
## vina_repulsion --> repulsion
atom_dict = {"6":"C","7":"N","8":"O","9":"F","15":"P","16":"S","17":"Cl","35":"Br","53":"I"}
change_name_dict = {}
for ft in feature_list:
    if "vina" in ft:
        change_name_dict[ft] = "_".join(ft.split("_")[1:])
    else:
        change_name_dict[ft] = f'{atom_dict[ft.split(".")[1]]}-{atom_dict[ft.split(".")[0]]}'
changed_feature_list = list(change_name_dict.values())

# 1. PLANet_Uw
dataset_name = 'PLANet_Uw'
npy_files = [str(p) for p in list(Path(f'{shap_dir}/{dataset_name}/').glob('*npy'))]
npy_files.sort()
Uw_shap_inter = np.concatenate([np.load(n) for n in npy_files])

Uw_train = pd.read_csv(f'/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/dataset/true_lig_alone/rm_core_ids/PLANet_Uw/complex_6A/train.csv', sep='\t', header=0, names=['unique_identify', '-logAffi'])
Uw_feature = pd.merge(Uw_train, features, on=['unique_identify','-logAffi'])
Uw_feature.rename(columns=change_name_dict, inplace=True)

if not Path.exists(Path(f'{shap_dir}/{dataset_name}/imgs')):
    Path.mkdir(Path(f'{shap_dir}/{dataset_name}/imgs'), exist_ok=True)

plt.figure(figsize=(19, 18), facecolor='w', edgecolor='k')
sns.set(font_scale=1.2)
shap.summary_plot(Uw_shap_inter, Uw_feature[changed_feature_list], plot_type="bar", show=False)
plt.suptitle(f'Training set: {dataset_name}', y=1.1)
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/Uw_inter_value_beeswarm.png", dpi=800, bbox_inches='tight')
plt.close()

cm = plt.cm.get_cmap('viridis_r')
mean_shap = np.abs(Uw_shap_inter).mean(0)
df = pd.DataFrame(mean_shap,index=changed_feature_list,columns=changed_feature_list)
df.where(df.values == np.diagonal(df), df.values*2, inplace=True)
plt.figure(figsize=(19, 18), facecolor='w', edgecolor='k')
sns.set(font_scale=1.2)
sns.heatmap(df,cmap=cm,fmt='.3g',cbar=True,vmin=0, vmax=0.5)
plt.yticks(rotation=0)
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/Uw_inter_value_heatmap.png", dpi=800, bbox_inches='tight')
plt.close()

mean_feature = np.mean(np.abs(np.sum(Uw_shap_inter, axis=1)), axis=0)
sorted_idx = np.argsort(mean_feature)
fig, ax = plt.subplots()
ax.barh(np.array(changed_feature_list)[sorted_idx][-10:], mean_feature[sorted_idx][-10:])
for i, v in enumerate(mean_feature[sorted_idx][-10:]):
    ax.text(v+0.025, i - .25, round(v, 3),
            color = 'blue', fontweight = 'bold')
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/Uw_inter_value_mean_shap_top_10.png", dpi=800, bbox_inches='tight')
plt.close()

mean_feature = np.mean(np.abs(np.sum(Uw_shap_inter, axis=1)), axis=0)
sorted_idx = np.argsort(mean_feature)
fig, ax = plt.subplots(figsize=(8,15))
ax.barh(np.array(changed_feature_list)[sorted_idx], mean_feature[sorted_idx])
for i, v in enumerate(mean_feature[sorted_idx]):
    ax.text(v+0.025, i - .25, round(v, 3),
            color = 'blue', fontweight = 'bold')
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/Uw_inter_value_mean_shap.png", dpi=800, bbox_inches='tight')
plt.close()

# 2. PDBbind_minimized
dataset_name = 'PDBbind_minimized'
npy_files = [str(p) for p in list(Path(f'{shap_dir}/{dataset_name}/').glob('*npy'))]
npy_files.sort()
Pm_shap_inter = np.concatenate([np.load(n) for n in npy_files])

Pm_train = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_rm_core_ids/complex_6A/train.csv', sep='\t', header=0, names=['unique_identify', '-logAffi'])
Pm_feature = pd.merge(Pm_train, features, on=['unique_identify','-logAffi'])
Pm_feature.rename(columns=change_name_dict, inplace=True)

if not Path.exists(Path(f'{shap_dir}/{dataset_name}/imgs')):
    Path.mkdir(Path(f'{shap_dir}/{dataset_name}/imgs'), exist_ok=True)

plt.figure(figsize=(19, 18), facecolor='w', edgecolor='k')
sns.set(font_scale=1.2)
shap.summary_plot(Pm_shap_inter, Pm_feature[changed_feature_list], plot_type="bar", show=False)
plt.suptitle(f'Training set: {dataset_name}', y=1.1)
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/{dataset_name}_inter_value_beeswarm.png", dpi=800, bbox_inches='tight')
plt.close()

cm = plt.cm.get_cmap('viridis_r')
mean_shap = np.abs(Pm_shap_inter).mean(0)
df = pd.DataFrame(mean_shap,index=changed_feature_list,columns=changed_feature_list)
df.where(df.values == np.diagonal(df), df.values*2, inplace=True)
plt.figure(figsize=(19, 18), facecolor='w', edgecolor='k')
sns.set(font_scale=1.2)
sns.heatmap(df,cmap=cm,fmt='.3g',cbar=True,vmin=0, vmax=0.5)
plt.yticks(rotation=0)
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/{dataset_name}_inter_value_heatmap.png", dpi=800, bbox_inches='tight')
plt.close()

mean_feature = np.mean(np.abs(np.sum(Pm_shap_inter, axis=1)), axis=0)
sorted_idx = np.argsort(mean_feature)
fig, ax = plt.subplots()
ax.barh(np.array(changed_feature_list)[sorted_idx][-10:], mean_feature[sorted_idx][-10:])
for i, v in enumerate(mean_feature[sorted_idx][-10:]):
    ax.text(v+0.025, i - .25, round(v, 3),
            color = 'blue', fontweight = 'bold')
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/{dataset_name}_inter_value_mean_shap_top_10.png", dpi=800, bbox_inches='tight')
plt.close()

mean_feature = np.mean(np.abs(np.sum(Pm_shap_inter, axis=1)), axis=0)
sorted_idx = np.argsort(mean_feature)
fig, ax = plt.subplots(figsize=(8,15))
ax.barh(np.array(changed_feature_list)[sorted_idx], mean_feature[sorted_idx])
for i, v in enumerate(mean_feature[sorted_idx]):
    ax.text(v+0.025, i - .25, round(v, 3),
            color = 'blue', fontweight = 'bold')
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{shap_dir}/{dataset_name}/imgs/{dataset_name}_inter_value_mean_shap.png", dpi=800, bbox_inches='tight')
plt.close()

# 3. PDBbind_minimized_intersected_Uw: not split
dataset_name = 'PDBbind_minimized_intersected_Uw'

PIP_npy = '/pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/shap_res/PDBbind_minimized_intersected_Uw/training_1.npy'
PIP_shap_inter = np.load(PIP_npy)

PIP_train = pd.read_csv('/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_intersected_Uw_rm_core_ids/complex_6A/train.csv', sep='\t', header=0, names=['unique_identify', '-logAffi'])
PIP_feature = pd.merge(PIP_train, features, on=['unique_identify','-logAffi'])
PIP_feature.rename(columns=change_name_dict, inplace=True)

out_dir = '/pubhome/xli02/project/PLIM/deep_learning/RFscore/3-shap/shap_res/PDBbind_minimized_intersected_Uw/imgs'
if not Path.exists(Path(out_dir)):
    Path.mkdir(Path(out_dir), exist_ok=True)

plt.figure(figsize=(19, 18), facecolor='w', edgecolor='k')
sns.set(font_scale=1.2)
shap.summary_plot(PIP_shap_inter, PIP_feature[changed_feature_list], plot_type="bar", show=False)
plt.suptitle(f'Training set: {dataset_name}', y=1.1)
plt.savefig(f"{out_dir}/{dataset_name}_inter_value_beeswarm.png", dpi=800, bbox_inches='tight')
plt.close()

cm = plt.cm.get_cmap('viridis_r')
mean_shap = np.abs(PIP_shap_inter).mean(0)
df = pd.DataFrame(mean_shap,index=changed_feature_list,columns=changed_feature_list)
df.where(df.values == np.diagonal(df), df.values*2, inplace=True)
plt.figure(figsize=(19, 18), facecolor='w', edgecolor='k')
sns.set(font_scale=1.2)
sns.heatmap(df,cmap=cm,fmt='.3g',cbar=True,vmin=0, vmax=0.5)
plt.yticks(rotation=0)
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{out_dir}/{dataset_name}_inter_value_heatmap.png", dpi=800, bbox_inches='tight')
plt.close()

mean_feature = np.mean(np.abs(np.sum(PIP_shap_inter, axis=1)), axis=0)
sorted_idx = np.argsort(mean_feature)
fig, ax = plt.subplots()
ax.barh(np.array(changed_feature_list)[sorted_idx][-10:], mean_feature[sorted_idx][-10:])
for i, v in enumerate(mean_feature[sorted_idx][-10:]):
    ax.text(v+0.025, i - .25, round(v, 3),
            color = 'blue', fontweight = 'bold')
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{out_dir}/{dataset_name}_inter_value_mean_shap_top_10.png", dpi=800, bbox_inches='tight')
plt.close()

mean_feature = np.mean(np.abs(np.sum(PIP_shap_inter, axis=1)), axis=0)
sorted_idx = np.argsort(mean_feature)
fig, ax = plt.subplots(figsize=(8,15))
ax.barh(np.array(changed_feature_list)[sorted_idx], mean_feature[sorted_idx])
for i, v in enumerate(mean_feature[sorted_idx]):
    ax.text(v+0.025, i - .25, round(v, 3),
            color = 'blue', fontweight = 'bold')
plt.title(f"Training set: {dataset_name}")
plt.savefig(f"{out_dir}/{dataset_name}_inter_value_mean_shap.png", dpi=800, bbox_inches='tight')
plt.close()


with open(f"/pubhome/xli02/project/PLIM/deep_learning/RFscore/test_res/{dataset_name}_Rm_core/1/best_model_VR1_RF.pkl", "rb") as f:
    Uw_model = joblib.load(f)
Uw_explainer = shap.TreeExplainer(Uw_model)
Uw_shap_exp = Uw_explainer(test_set[feature_list])