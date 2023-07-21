from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw, AllChem
from rdkit import DataStructs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

if not Path('MW_pAffi').exists():
    Path('MW_pAffi').mkdir()

if not Path('SASA_pAffi').exists():
    Path('SASA_pAffi').mkdir()

# 1. whole set
index_dir = '/pubhome/xli02/project/PLIM/v2019_dataset/index'
PIP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset.csv', sep='\t')
PLANet_df = pd.read_csv(f'{index_dir}/For_ML/20220524_other_files/PLANet_Uw_dealt_median.csv', sep='\t')
PIPUP_df = pd.read_csv(f'{index_dir}/For_ML/PDBbind_subset_union_BindingNet.csv', sep='\t')


# MW/SASA_pAffi
all_prop_sasa_pfam = pd.read_csv('../../3-property_calculate/2-other_property/all_prop_sasa_pfam.csv', sep='\t')
len(all_prop_sasa_pfam) # 86978

PIP_prop = all_prop_sasa_pfam[all_prop_sasa_pfam['unique_identify'].isin(PIP_df['pdb_id'])].copy()
PLANet_prop = all_prop_sasa_pfam[all_prop_sasa_pfam['unique_identify'].isin(PLANet_df['unique_identify'])].copy()
PIPUP_prop = all_prop_sasa_pfam[all_prop_sasa_pfam['unique_identify'].isin(PIPUP_df['unique_identify'])].copy()
PIP_prop.reset_index(inplace=True, drop=True)
PLANet_prop.reset_index(inplace=True, drop=True)
PIPUP_prop.reset_index(inplace=True, drop=True)

## BindingNet: MW
pearsonr = round(stats.pearsonr(PLANet_prop['mw'], PLANet_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PLANet_prop['mw'], PLANet_prop['-logAffi'])[0],3)

xy = np.vstack([PLANet_prop['mw'], PLANet_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PLANet_prop['mw'][idx], PLANet_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('Molecular Weight (Da)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,1000)
ax.set_ylim(0,16)
ax.set_title(f'BindingNet (N={len(PLANet_df)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('MW_pAffi/PLANet_Mw_pAffi_density_scaled_1000.png', dpi=300, bbox_inches='tight')
plt.close()


## BindingNet: SASA
pearsonr = round(stats.pearsonr(PLANet_prop['del_sasa'], PLANet_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PLANet_prop['del_sasa'], PLANet_prop['-logAffi'])[0],3)

xy = np.vstack([PLANet_prop['del_sasa'], PLANet_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PLANet_prop['del_sasa'][idx], PLANet_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,18)
ax.set_ylim(0,15)
ax.set_title(f'BindingNet (N={len(PLANet_df)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('SASA_pAffi/PLANet_SASA_pAffi_density_scaled_18.png', dpi=300, bbox_inches='tight')
plt.close()

## PDBbind_subset: MW
pearsonr = round(stats.pearsonr(PIP_prop['mw'], PIP_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PIP_prop['mw'], PIP_prop['-logAffi'])[0],3)

xy = np.vstack([PIP_prop['mw'], PIP_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PIP_prop['mw'][idx], PIP_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('Molecular Weight (Da)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,1000)
ax.set_ylim(0,16)
ax.set_title(f'PDBbind_subset (N={len(PIP_prop)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('MW_pAffi/PDBbind_Mw_pAffi_density_scaled_1000.png', dpi=300, bbox_inches='tight')
plt.close()


## PDBbind_subset: SASA
pearsonr = round(stats.pearsonr(PIP_prop['del_sasa'], PIP_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PIP_prop['del_sasa'], PIP_prop['-logAffi'])[0],3)

xy = np.vstack([PIP_prop['del_sasa'], PIP_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PIP_prop['del_sasa'][idx], PIP_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,18)
ax.set_ylim(0,15)
ax.set_title(f'PDBbind_subset (N={len(PIP_prop)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('SASA_pAffi/PDBbind_SASA_pAffi_density_scaled_18.png', dpi=300, bbox_inches='tight')
plt.close()

## PIPUP: MW
pearsonr = round(stats.pearsonr(PIPUP_prop['mw'], PIPUP_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PIPUP_prop['mw'], PIPUP_prop['-logAffi'])[0],3)

xy = np.vstack([PIPUP_prop['mw'], PIPUP_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PIPUP_prop['mw'][idx], PIPUP_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('Molecular Weight (Da)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,1000)
ax.set_ylim(0,16)
ax.set_title(f'PDBbind_subset' +r'$\cup$' f'BindingNet (N={len(PIPUP_df)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('MW_pAffi/PIPUP_Mw_pAffi_density_scaled_1000.png', dpi=300, bbox_inches='tight')
plt.close()

## PIPUP: SASA
pearsonr = round(stats.pearsonr(PIPUP_prop['del_sasa'], PIPUP_prop['-logAffi'])[0],3)
spearmanr = round(stats.spearmanr(PIPUP_prop['del_sasa'], PIPUP_prop['-logAffi'])[0],3)

xy = np.vstack([PIPUP_prop['del_sasa'], PIPUP_prop['-logAffi']])  #按行叠加
g = gaussian_kde(xy)  #根据xy进行核密度估计(kde) -> 关于xy的概率密度函数
z = g(xy)  #计算每个xy样本点的概率密度

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()  #对z值排序并返回索引
y, y_, z = PIPUP_prop['del_sasa'][idx], PIPUP_prop['-logAffi'][idx], z[idx]  #对y, y_根据z的大小进行排序

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(y, y_, s=2, c=z, zorder=2)
ax.set_xlabel('buried SASA (nm$\mathregular{^2}$)', fontsize=15)
ax.set_ylabel('Experimental pAffi', fontsize=15)
ax.tick_params(axis='x', labelsize= 12)
ax.tick_params(axis='y', labelsize= 12)
ax.set_xlim(0,18)
ax.set_ylim(0,15)
ax.set_title(f'PDBbind_subset' +r'$\cup$' f'BindingNet (N={len(PIPUP_df)})\nRp={pearsonr},Rs={spearmanr}', fontsize=15)
plt.savefig('SASA_pAffi/PIPUP_SASA_pAffi_density_scaled_18.png', dpi=300, bbox_inches='tight')
plt.close()

