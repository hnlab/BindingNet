# BindingNet Database
# 1. Enviroment
- build database: `conda env create -f bindingnet_generate.yml`
## 2. Workflow
## 2.0 Download PDBbind_v2019 dataset
## 2.1 Extract PDB ID in PDBbind
- `cd 1-extract_PDBid`
- `bash extractPDBid.sh` -> `PDBIDs_INDEX_general_PL_data.2019`
## 2.2 Map PDB ID to Uniprot ID
- [Retrieve/ID mapping tool](https://www.uniprot.org/uploadlists/)
- Upload the file `PDBIDs_INDEX_general_PL_data.2019`
- select from `PDB` to `UniPortKB` in Select options and click `Submit`
- Colunms: `Your list...(PDB ID)`, `Entry`, `ChEMBL`, (**`BindingDB`**), `Protein names`
- Download: `Tab-seperated` -> `2-query_target_ChEMBLid/converted_PDBIDs_INDEX_general_PL_data.2019.tab`
- change the name of column #1 to `PDB ID`
- run query `select * where a3 !== ""` in RBQL Console (VSCode extension Rainbow CSV)
    - `Ctrl` + `Shift` + `P` at VSCode
    - `Rainbow CSV: RBQL`
    - `select * where a3 !== ""` -> `2-query_target_ChEMBLid/converted_PDBIDs_INDEX_general_PL_data.2019.tab.tsv`
## 2.3 Search corresponding ligand for each receptor which has unique `UniPort ID` from ChEMBL
- `cd 3-query_ChEMBL`
- `python query_chembl_v2019_x019.py`
  - ChEMBL database in lab
  - [`ChEMBL webresource client`](https://github.com/chembl/chembl_webresource_client)
    - support multithreaded for high I/O
## 2.4 Find compounds similar with crystal ligand within each target
- `cd 4-extract_similar_compnds`
- `bash extract_simi_compounds.sh`
## 2.5 Align - Filter Serious Clash - Rescore - Filter by energy - Calculate core RMSD - Extract final pose - Add Affinity
- `cd 5-pipeline_after_4-extract_simi_compnds`
### 2.5.1 obtain all target_pdbid_compound list
- `cd 1-obtain_list`
- `bash obtain_target_pdbid_list.sh` -> `all_target_pdbid.list` -> for rec_opt
- `bash obtain_target_pdbid_compound_list.sh` -> `all_target_pdbid_compound.list`
### 2.5.2 receptor h `opt yes` in plop6.0
- `cd 2-rec_opt`
- `bash rec_opt_qsub_anywhere.sh`
### 2.5.3 align - filter Serious Clash - rescore - filter by energy - calculate core RMSD - extract final pose
- `cd 3-align-filter_clash-rescore-final_filter`
- `qsub -p -100 run_for_each_compound.sh`
- task array: `-t start-end`
  - `${SGE_TASK_ID}`: "target pdbid_compound_id"
  - **less than 75000 tasks** once
    - split 195686 tasks into 3 fold
    - 70,000 tasks for each script
### 2.5.4 Output
- rec_h_opt.pdb
- CHEMBLxxx_xxxx_final.csv
- CHEMBLxxx_xxxx_dlig_xxx_dtotal_xxxCoreRMSD_xxx_ene.csv
- CHEMBLxxx_xxxx_dlig_xxx_dtotal_xxxCoreRMSD_xxx_final.pdb
## 2.6 Deal with result: generate PLANet_all/PLANet_Uw index
- `cd 5-pipeline_after_4-extract_simi_compnds/4-deal_with_result`
## 2.7 Requery And Obtain all activities for SAR
- `cd 5-pipeline_after_4-extract_simi_compnds/5-Requery_And_Obtain_all_affinity_for_SAR`
## 2.8 Convert `_final.pdb` to `compound.sdf`, and Extract pocket for machine learning
- `cd 5-pipeline_after_4-extract_simi_compnds/6-convert_sdf_AND_extract_pocket`
## 2.9 PDBbind minimized
- `cd 5-pipeline_after_4-extract_simi_compnds/7-PDBbind_v2019_minimize`
## 2.10 Train on SGCNN model
- `cd 6-deep_learning/2-FAST/`
## 2.11 Construct web_server
- `cd 7-web_server/`
## 2.12 Analysis property distribution
- `cd 10-analysis`