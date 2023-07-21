# 6.17-18
[reference code](https://github.com/hnlab/can-ai-do/tree/master/tutorial)
## 1. Setup conda environment
- for cpu:
    - `conda create -n acnn -c deepchem -c rdkit -c conda-forge -c omnia deepchem=2.3.0 tqdm seaborn rdkit scikit-learn parallel openbabel python=3.6`
        - 指定多个channel, 优先级从左至右
    - 但是在`import deepchem`的时候会报错：
        - `from sklearn.metrics import jaccard_similarity_score`   
        `ImportError: cannot import name 'jaccard_similarity_score'`
        - 是由于新版本的`sklearn`中没有该模块
        - 将`sklearn`的版本退回到`0.22`，**同时要不改变其他包如deepchem的版本**
            - `conda install -c deepchem -c rdkit -c conda-forge -c omnia deepchem=2.3.0 tqdm seaborn rdkit scikit-learn=0.22 parallel openbabel python=3.6`
    - jupyter notebook中没有显示该kernel
        - `python -m ipykernel install --user --name acnn --display-name "Python acnn"`
        - 重启窗口
- for gpu:
    - `conda create -n acnn -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.3.0 tqdm seaborn rdkit scikit-learn=0.22 parallel openbabel python=3.6`
## 2. ACNN on PDBbind_v2015
### 2.1 download PDBbind v2015
- `bash get_pdbbind.sh`
- `python ACNN.py -component ligand -subset refined -test_core 2015 -result_dir result/ligand_refined_core_sequence > pdbbind_train_refine_test_core_lig_alone.log`
### 2.2 read scripts
- pdbbind_datasets.py
    - 根据路径提取ligand.pdb, pocket.pdb/protein.pdb, 活性(label)
    - `get_shards()`: 若数据量过大，分批处理
    - 提取特征
        - 返回（提取成功的特征量，提取失败的index）   
            (lig_mol, lig_coords, lig_neighbor_list, lig_z),    
            (pock_mol, pock_coords, pock_neighbor_list, pock_z),    
            (complex_mol, complex_coods, complex_neighbor_list, complex_z)
        - `lig_z`: `get_Z_matrix()`, 返回**原子序数**
        - `lig_neighbor_list`: `compute_neighbor_list()`
            - 通过`mdtraj.Trajectory()`, `mdtraj.geometry.compute_neighborlist()`获得neighbor_list ?
    - 存储特征化后数据集: `DiskDataset.create_dataset()` / load本地特征化后的数据集： `deepchem.data.DiskDataset(feat_dir)`
        - `feat_dir`下含有的数据
            - `shard-i-X.joblib`: 输入数据，即提取的特征量
            - `shard-i-y.joblib`：lable（输出数据），即活性数据
            - `shard-i-w.joblib`：权重
            - `shard-i-ids.joblib`：每个样本的唯一标识符，如pdbid
            - `task.json`: 由标签列的列名组成
            - `metadata.csv.gzip`: 所有 X, y, w, ids组成的dataframe
    - Split dataset: `splitter.train_valid_test_split()`
        - **比例: 默认8:1:1**
    - `sort_inds = np.argsort(labels)`: 根据活性排序，活性接近的放在一起，能train得更好？
- atomic_conv.py
# 6.29
## 1. 跑ligand alone
- 参数
```python
subset = 'PLIM_dataset'
version = '1'
data_dir = '/home/xli/Documents/projects/ChEMBL-scaffold/v2019_dataset'

component = 'ligand'

shard_size=4096
# shard_size=1024

#for protein
load_binding_pocket=False
frag2_num_atoms = 24000
```
- `python ACNN.py -component ligand -result_dir result/ligand_alone_random`
    - 162,295 有活性数据/ 162,323 有'final.pdb'/ 195,677 已完成
    - 内存过大？ -> Killed -> 用~~GPU？~~(30/60 G, honda上也是60G) / pocekt.pdb?
        - `shard_size=1024`: 在运行到第7个shard时也会被Kill
        - 用某个小的pdb文件替换`protein.pdb`
            - 在Chimera上画一个苯 -> `false_rec.pdb`
    - 在创建完数据集后中断 -> `nohup python ACNN.py -component ligand -result_dir result/ligand_alone_random -reload > plim_ligand_alone_nu_2.log &`
    - 在第3/4个epoch时，`computed_metrics`('pearson_r2_score' & 'mean_absolute_error')均为`nan`
        - 在将遇到`nan`后的行为由`break`改为`continue`后，之后的`computed_metrics`全为`nan`
        - 应该是模型的问题 ——换模型
- 取前10000数据尝试: `nohup python ACNN.py -component ligand -result_dir result/top_10000_ligand_alone_random > plim_ligand_alone_nu_4.log &`
    - `feat_dir`
    - `truncated_df`
    - 成功
- 取前/后80000数据尝试
    - memory error? / Killed
    - top80000: nan
```bash
#需修改输出目录以及log文件名称
# nohup python ACNN.py \
#     -component ligand \
#     -dataset_end_idx 80000 \
#     -result_dir result/top_80000_ligand_alone_random \
#     > plim_ligand_alone_nu_top80000.log &
# qsub_anywhere.py -c 'conda activate acnn; python /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/ACNN.py -component ligand -dataset_end_idx 80000 -result_dir /home/xli/Documents/projects/ChEMBL-scaffold/deep_learning/acnn_can_ai_do/result/top_80000_ligand_alone_random' -j . -q honda -N 'ACNN_top8w' --qsub_now
qsub acnn_top4w.sh
qsub acnn_tail.sh
```
- CPU VS **GPU**
    - CPU更稳定
    - GPU更快
- 验证是否跑了GPU： `nvidia-smi`
# 7.14
## 1. cuda队列上
- `Succeeded to featurize 0/40000 samples.` -> `IndexError: index 0 is out of bounds for axis 0 with size 0`
    - `false_rec.pdb`没有更改路径
# 7.27
## 1. `Killed`问题
- 经查看log及[ganglia](https://www.huanglab.org.cn/ganglia/?c=b060cuda.hn.org&m=load_one&r=hour&s=by%20name&hc=4&mc=2)
    - ![Screenshot from 2021-07-27 14-58-31](https://user-images.githubusercontent.com/54713559/127109939-0736f2dd-914e-4ad3-9360-7341f658a95e.png)
    - 在对每个shard提取特征量和`create_dataset`时内存均达到一个峰值，在到第25个shard时，内存超过`Mem` + `Swap`的总值 -> 被Killed
- 在队列上，若内存需求量大，可以占满整个节点：如`#$ -pe cuda 4`
- 在cuda上，`Mem`+`Swap` ~ 64G; 在ampere上, ~70G
- [在benz上](https://github.com/hnlab/Cluster_Manual_mly/blob/main/dell_cluster.md#opel-mazda-%E5%92%8C-benz-%E9%98%9F%E5%88%97)：k151-156 为 256G， 但为CPU
    - 可尝试在benz上load dataset，再转移至GPU节点上进行训练
- k228: 62 + 127 ~ 190G
- [ ] PLIM_all_lig_alone: 最大需124G；**nan值**
    - 些许不同的结构（如突变），活性相同？confuse?
- [x] PDBbind * 2
- [x] PLIM_unique_lig_alone within target
- [x] PLIM_unique_lig_alone within target remove outlier(-logAffi < 2)
    - `python plot.py -result_dir unique_ligand_alone_random_removed_outlier_patience_5 -subset train -log_name uniq_lig_alone_removed_outlier_patience_5`
- [ ] PLIM_unique_lig_alone cross target
    - **at benz**
- [ ] PLIM_unique_complex
    - 跑到第5个shard时，10h内没有进展 -> `ps xf` -> `STAT`:`Sl+` -> sleep
    - **at benz**
    - [ ] pocket 6A
- [x] reproduce: cluster_file
- [x] shuffle_batches
- [ ] PDBbind + PLIM_unique
    - 逐渐增加最不相似的compound
        - **`mmp.LazyBitVectorPick`**
    - tail_8w + PDBbind
- [x] PDBbind cry_lig minimize
- [ ] R2, pearsonr, spearmanr, [mae, mse/rmse](https://zhuanlan.zhihu.com/p/37663120)
    - mae(mean_absolute_error)：平均绝对误差，用于评估预测结果和真实数据集的接近程度的程度，其其值越小说明拟合效果越好   
    ![Screenshot from 2021-07-28 11-39-24](https://user-images.githubusercontent.com/54713559/127259868-e6ad277c-94f4-4c93-a1ac-c1e5e936c2db.png)
    - mse(mean_squared_error)：均方差，该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好   
    ![Screenshot from 2021-07-28 11-38-35](https://user-images.githubusercontent.com/54713559/127259824-89040801-989c-4df5-8123-57e5a039e1b9.png)
    - R2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差

