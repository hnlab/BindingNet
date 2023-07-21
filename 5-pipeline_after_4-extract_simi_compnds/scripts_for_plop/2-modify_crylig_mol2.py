'''
modify mol2 file for generating parameter
add altLoc colunmn
'''
import sys

mol2_file = sys.argv[1]
modified_mol2_file = sys.argv[2]

def modify_mol2(mol2_file, modified_mol2_file):
    with open(mol2_file, 'r') as f1:
        lines = f1.readlines()
    
    with open(modified_mol2_file, 'w', newline='') as f2:
        an_too_long_ids=0
        for line in lines:
            newline = line
            if ' LIG ' in line:
                atom_name_original = line.split()[1]
                additional_num=0
                if 'UNK_' in atom_name_original:
                    print('"UNK_" is in mol2 file, modify it.')
                    additional_num = len(atom_name_original)-5
                atom_id = line.split()[0]
                atom_type = line.split()[5].split('.')[0]
                if len(atom_type+atom_id) > 4:
                    an_too_long_ids = an_too_long_ids+1
                    atom_id = f'{an_too_long_ids:02}'
                    print(f'length of atom_name > 4, modify its "atom_id" to padding format: {atom_type+line.split()[0]} -> {atom_type+atom_id}.')
                if len(atom_type) == 1:
                    atom_name = atom_type + f'{atom_id:3}' + " "*(additional_num+1)
                    newline = line[0:5] + atom_name + line[(10+additional_num):]
                else: # atom_type = "CL"...
                    atom_name = atom_type.upper() + f'{atom_id:3}' + " "*(additional_num)
                    newline = line[0:5] + atom_name + line[(10+additional_num):]
            f2.write(newline)

modify_mol2(mol2_file, modified_mol2_file)
