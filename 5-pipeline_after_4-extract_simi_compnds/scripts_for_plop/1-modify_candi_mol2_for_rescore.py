'''
modify mol2 file for rescoring
add Name; add altLoc colunmn
'''
import sys

filtered_mol2_file = sys.argv[1]
filtered_modified_mol2_file = sys.argv[2]

def modify_mol2(filtered_mol2_file, filtered_modified_mol2_file):
    with open(filtered_mol2_file, 'r') as f1:
        lines = f1.readlines()
    
    start_index = end_index = 0
    with open(filtered_modified_mol2_file, 'w', newline='') as f2:
        for i in range(lines.count("@<TRIPOS>MOLECULE\n")):
            if "@<TRIPOS>MOLECULE\n" in lines[start_index + 1:len(lines)]:
                end_index = lines.index("@<TRIPOS>MOLECULE\n", start_index + 1) - 1
            else:
                end_index = len(lines)      #
            COMPND_ID = f"{lines[start_index+1].split('_')[0]}-{lines[start_index+1].split('_')[2]}-{lines[start_index+1].split('_')[4]}"
            f2.write(f"##########                 Name:     {COMPND_ID}\n")
            for line in lines[start_index:end_index]:      #remove empty line and "@<TRIPOS>SUBSTRUCTURE"
                newline = line
                if 'LIG' in line:
                    atom_id = line.split()[0]
                    atom_name = line.split()[5].split('.')[0]
                    if len(atom_name) == 1:
                        newline = line[0:5] +  atom_name + f'{atom_id:3}' + " "+ line[10:]
                    else: # atom_name = "CL"...
                        newline = line[0:5] +  atom_name.upper() + f'{atom_id:3}' + line[10:]
                f2.write(newline)
            start_index = end_index + 1

modify_mol2(filtered_mol2_file, filtered_modified_mol2_file)

