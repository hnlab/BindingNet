import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

pharos_f = 'from_pharos_tcrd/pharos_query_20220215.csv'
pharos_df = pd.read_csv(pharos_f)
pharos_df.drop(columns=['id'], inplace=True)

uniprot_chembl_f = 'converted_PDBIDs_INDEX_general_PL_data.2019.tab.tsv'
# uniprot_chembl_df = pd.read_csv(uniprot_chembl_f, sep='\t')
pdbid2uniprot={}
header = True
with open(uniprot_chembl_f, 'r') as f:
    lines = f.readlines()
for line in lines:
    if header:
        header = False
        continue
    pdbids = line.rstrip().split('\t')[0]
    uniprotid = line.rstrip().split('\t')[1]
    for pdbid in pdbids.split(','):
        if pdbid in pdbid2uniprot.keys():
            va = pdbid2uniprot[pdbid]
            pdbid2uniprot[pdbid] = va + ',' + uniprotid
        else:
            pdbid2uniprot[pdbid] = uniprotid

pdbid_uniprotid_df = pd.DataFrame.from_dict(pdbid2uniprot, orient='index', columns=['UniProt']).reset_index().rename(columns={'index':'Cry_lig_name'})
pdbid_uniprotid_df['UniProt'] = pdbid_uniprotid_df['UniProt'].map(lambda x: x.split(','))
pdbid_uniprotid_df_exploded = pdbid_uniprotid_df.explode('UniProt') #一个pdbid(共889/12787个)对应多个Uniprot ID时，都加入count中

pdbid_uniprot_family_df = pd.merge(pdbid_uniprotid_df_exploded, pharos_df, on='UniProt', how="left")
PDBbind_mapped_notna_df = pdbid_uniprot_family_df[pdbid_uniprot_family_df['Family'].notna()]

# Uw
Uw_dealt_median = f'/pubhome/xli02/project/PLIM/v2019_dataset/index/PLANet_Uw_dealt_median.csv'
Uw_dealt_median_df = pd.read_csv(Uw_dealt_median, sep='\t')
Uw_mapped_df = pd.merge(Uw_dealt_median_df, PDBbind_mapped_notna_df, on='Cry_lig_name', how="left")
Uw_mapped_df.loc[Uw_mapped_df['UniProt'].isna(), 'Family'] = 'Unclassified'

# labels = [x for x in set(Uw_mapped_df['Family']) if len(Uw_mapped_df[Uw_mapped_df['Family']==x])/len(Uw_mapped_df) >= 0.01]
labels = list(set(Uw_mapped_df['Family']))
sizes = [len(Uw_mapped_df[Uw_mapped_df['Family']==x]) for x in labels]
dictionary = dict(zip(labels, sizes))
sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
x = np.char.array(list(sorted_dictionary.keys()))
y = np.array(list(sorted_dictionary.values()))
percent = 100.*y/sum(y)
labels_2 = ['{0}  {1:1.2f}%'.format(i,j) for i,j in zip(x, percent)]
cs = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:green', 'tab:olive', 'tab:red', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:orange']

patches, texts = plt.pie(sorted(sizes, reverse=True), startangle=90, colors=cs, counterclock=False, wedgeprops={'edgecolor': 'w', 'linewidth':1})
plt.legend(patches, labels_2, loc="center", prop={'size':5})
plt.title(f'Target classify on PLANet_Uw')
center_circle = plt.Circle((0,0), 0.7, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)
plt.savefig('from_pharos_tcrd/Uw_target_classify.png', dpi=300, bbox_inches='tight')
Uw_mapped_df.to_csv('from_pharos_tcrd/PLANet_Uw_dealt_median_target_classify.csv', sep='\t', index=False)

# PLANet_all
all_dealt_median = f'/pubhome/xli02/project/PLIM/v2019_dataset/index/PLANet_all_dealt_median.csv'
all_dealt_median_df = pd.read_csv(all_dealt_median, sep='\t')
all_mapped_df = pd.merge(all_dealt_median_df, PDBbind_mapped_notna_df, on='Cry_lig_name', how="left")
all_mapped_df.loc[all_mapped_df['UniProt'].isna(), 'Family'] = 'Unclassified'

labels = list(set(all_mapped_df['Family']))
sizes = [len(all_mapped_df[all_mapped_df['Family']==x]) for x in labels]
dictionary = dict(zip(labels, sizes))
sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
x = np.char.array(list(sorted_dictionary.keys()))
y = np.array(list(sorted_dictionary.values()))
percent = 100.*y/sum(y)
labels_2 = ['{0}  {1:1.2f}%'.format(i,j) for i,j in zip(x, percent)]
cs = ['tab:blue', 'tab:cyan', 'tab:orange', 'tab:green', 'tab:olive', 'tab:red', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:orange']

patches, texts = plt.pie(sorted(sizes, reverse=True), startangle=90, colors=cs, counterclock=False, wedgeprops={'edgecolor': 'w', 'linewidth':1})
plt.legend(patches, labels_2, loc="center", prop={'size':5})
plt.title(f'Target classify on PLANet_all')
center_circle = plt.Circle((0,0), 0.7, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)
plt.savefig('from_pharos_tcrd/PLANet_all_target_classify.png', dpi=300, bbox_inches='tight')
all_mapped_df.to_csv('from_pharos_tcrd/PLANet_all_dealt_median_target_classify.csv', sep='\t', index=False)