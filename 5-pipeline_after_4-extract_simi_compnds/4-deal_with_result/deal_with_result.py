import glob
from pathlib import Path
import pandas as pd
import numpy as np

data_dir = '/home/xli/git/BindingNet/v2019_dataset'
target_dirs = glob.glob(data_dir + '/web*/', recursive=False)
all_compnd_dirs = []
for target_dir in target_dirs:
    pdb_dirs = glob.glob(target_dir + '/C*/', recursive=False)
    for pdb_dir in pdb_dirs:
        compnd_dirs = glob.glob(pdb_dir + '/C*/', recursive=False)
        all_compnd_dirs = all_compnd_dirs + compnd_dirs

final_done_list = []
error_code_list = []
with_final_pdb_list = []
all_df = []
n = 0
for all_compnd_dir in all_compnd_dirs:
    n = n + 1
    whole_log_text = Path(f'{all_compnd_dir}/whole.log').read_text()
    if "FINAL DONE" in whole_log_text:
        final_done_list.append(all_compnd_dir)
    if "error_code" in whole_log_text:
        error_code_list.append(all_compnd_dir)
    final_pdbs = glob.glob(f'{all_compnd_dir}/*final.pdb', recursive=False)
    if len(final_pdbs) != 0:
        with_final_pdb_list.append(final_pdbs[0])
        final_csvs = glob.glob(f'{all_compnd_dir}/*final.csv', recursive=False)
        if len(final_csvs) != 0:
            final_df = pd.read_csv(final_csvs[0], sep="\t",dtype={'Cry_lig_name':str})
            all_df.append(final_df)
    if n%1000 == 0:
        print(f'Compelete {n}.')

all_df = pd.concat(all_df, ignore_index = True)
all_df.to_csv(f'{data_dir}/index/BindingNet_dataset_v1_all.csv', sep = "\t", index = False)

def modify_pdb_format():
    compnd_wrong_format = []
    for idx, cpnd in enumerate(with_final_pdb_list):
        with open(cpnd, 'r') as f:
            lines = f.readlines()
        with open(cpnd, 'w') as f:
            for line in lines:
                newline = line
                if 'ATOM' in line:
                    atom_name = line.split()[-1]
                    if len(atom_name) != 1:
                        # print(len(line.replace(f' {atom_name}', f'{atom_name}')))
                        compnd_wrong_format.append(cpnd)
                        newline = line[:76] + line[77:]
                f.write(newline)
        if idx % 1000 == 0:
            print(f'Complete {idx}.')
    # len(compnd_wrong_format) #59786

def fix_error_of_ugmL():
    '''
    Fix the error of ug/mL
    '''
    modified_df = all_df.copy()
    modified_list = []
    for i,row in enumerate(modified_df.itertuples()):
        if row.Affinity == 'No data':
            continue
        affi_file = f'{data_dir}/web_client_{row.Target_chembl_id}/web_client_{row.Target_chembl_id}-activity.tsv'
        affi_df = pd.read_csv(affi_file, sep="\t")
        filter_units_affi_df = affi_df.query('standard_units in ["nM", "ug.mL-1"]').copy()
        affi_rows = filter_units_affi_df[filter_units_affi_df['molecule_chembl_id'] == row.Similar_compnd_name]

        if 'Kd' in set(affi_rows['standard_type']):
            filtered_rows = affi_rows.query('standard_type == "Kd"')
        elif 'Ki' in set(affi_rows['standard_type']):
            filtered_rows = affi_rows.query('standard_type == "Ki"')
        elif 'IC50' in set(affi_rows['standard_type']):
            filtered_rows =  affi_rows.query('standard_type == "IC50"')
        else:
            filtered_rows = affi_rows.query('standard_type == "EC50"')

        if '=' in set(filtered_rows['standard_relation']):
            relation_filtered_rows = filtered_rows.query('standard_relation == "="').copy()
        else:
            # print(f"{row.Target_chembl_id}_{row.Similar_compnd_name} only has affinity " \
            #     f"data with relation '<', use it as '='.")
            relation_filtered_rows = filtered_rows.query('standard_relation == "<"').copy()

        for relation_row in relation_filtered_rows.itertuples():
            if relation_row.standard_units == 'ug.mL-1':
                # relation_filtered_rows.loc[relation_row.Index, 'standard_value'] = relation_row.standard_value * row.MolWt * 1000
                relation_filtered_rows.loc[relation_row.Index, 'standard_value'] = relation_row.standard_value*1000000/row.MolWt 
                relation_filtered_rows.loc[relation_row.Index, 'standard_units'] = 'nM'

        best_affi_row = relation_filtered_rows.sort_values(by='standard_value').head(1)
        # if len(relation_filtered_rows) > 1:
        #     print(f"{row.Target_chembl_id}_{row.Similar_compnd_name} has " \
        #         f"{len(relation_filtered_rows)} different " \
        #         f"{best_affi_row['standard_type'].values[0]} affinity data, " \
        #         f"choose the best affinity data among them.")

        affi = f"{best_affi_row['standard_type'].values[0]} " \
            f"{best_affi_row['standard_relation'].values[0]} " \
            f"{best_affi_row['standard_value'].values[0]} " \
            f"{best_affi_row['standard_units'].values[0]}"
        if modified_df.loc[row.Index, 'Affinity'] != affi:
            modified_list.append(f'{row.Target_chembl_id}_{row.Similar_compnd_name}')
        modified_df.loc[row.Index, 'Affinity'] = affi
        modified_df.loc[row.Index, 'Activity_id'] = best_affi_row['activity_id'].values[0]
        if i%1000 == 0:
            print(f'Complete {i}/{len(modified_df)}.')
    modified_df['Activity_id'] = modified_df['Activity_id'].astype(int)
    modified_df.to_csv(f'{data_dir}/index/BindingNet_dataset_v1_all_modified.csv', sep = "\t", index = False)
    return modified_df

modified_df = fix_error_of_ugmL()

# Fix the found error of ChEMBL dataset manually
# Affinity of paper in CHEMBL1135542(part of), CHEMBL1144810, CHEMBL1140026 and CHEMBL1137929 are wrong
# corrosponding to target CHEMBL205, CHEMBL1907, CHEMBL1827, and CHEMBL3589

# Deal with 'all.csv'
remove_no_affinity_modified = modified_df[modified_df['Affinity'] != 'No data'].copy()
remove_no_affinity_modified['-logAffi'] = [-np.log10(float(affi.split(' ')[2]) * 10 ** -9) for affi in remove_no_affinity_modified['Affinity']]
remove_no_affinity_modified['unique_identify'] = [f'{row.Target_chembl_id}_{row.Cry_lig_name}_{row.Similar_compnd_name}' for row in remove_no_affinity_modified.itertuples()]
modified_dealt_csv = f'{data_dir}/index/BindingNet_dataset_v1_deal_affi_modified.csv'
remove_no_affinity_modified.to_csv(modified_dealt_csv, sep = "\t", index = False)

final_modified_df = remove_no_affinity_modified[['unique_identify', '-logAffi']]
final_modfied_csv = f'{data_dir}/index/BindingNet_dataset_v1_final_modified.csv'
final_modified_df.to_csv(final_modfied_csv, sep = "\t", index = False)

# Unique compound within target
remove_no_affinity_modified['target_cpx'] = [f'{uniq_identi.split("_")[0]}_{uniq_identi.split("_")[2]}' for uniq_identi in remove_no_affinity_modified['unique_identify']]
# len(remove_no_affinity_modified['target_cpx'].unique())   # 71060
grouped = remove_no_affinity_modified.groupby('target_cpx')
unique_target_cpx_best_total_delta_df = grouped.apply(lambda x: x.sort_values(by=['Total_delta', 'Lig_delta', 'Core_RMSD'], ascending=[True, False, True]).head(1))
unique_file = f'{data_dir}/index/BindingNet_dataset_v1_deal_affi_unique_modified.csv'
unique_target_cpx_best_total_delta_df.to_csv(unique_file, sep = "\t", index = False)

unique_simple_df = unique_target_cpx_best_total_delta_df[['unique_identify', '-logAffi']]
unique_simple_csv = f'{data_dir}/index/BindingNet_dataset_v1_final_unique_modified.csv'
unique_simple_df.to_csv(unique_simple_csv, sep = "\t", index = False)
