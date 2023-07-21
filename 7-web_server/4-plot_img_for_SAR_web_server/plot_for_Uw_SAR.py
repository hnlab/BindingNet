from matplotlib import pyplot as plt
import seaborn as sns
from  matplotlib import cm
import numpy as np
import pandas as pd

wrkdir = '/home/lixl/Documents/Project/ChEMBL-scaffold/12-web_server'
Uw_for_SAR_info = f'{wrkdir}/1-mk_table/PLANet_Uw_for_SAR_rm_NA_unique.csv'
Uw_for_SAR_info_df = pd.read_csv(Uw_for_SAR_info, sep='\t') #105920

# target_classification
target_pdbid_uniprot_mapped_pharos = pd.read_csv(f'{wrkdir}/1-mk_table/planet_Uw/pdbid_mapped_uniprot_family_chembl_info.csv', sep='\t') #6032
target_pdbid_uniprot_mapped_pharos.rename(columns={'pdbid':'Cry_lig_name', 'target_chembl_id':'Target_chembl_id'}, inplace=True)
Uw_target_classify = pd.merge(Uw_for_SAR_info_df, target_pdbid_uniprot_mapped_pharos, on=['Target_chembl_id', 'Cry_lig_name'], how='left') #106472

labels = list(set(Uw_target_classify['target_family_from_pharos']))
sizes = [len(Uw_target_classify[Uw_target_classify['target_family_from_pharos']==x]) for x in labels]
dictionary = dict(zip(labels, sizes))
sorted_dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
x = np.char.array(list(sorted_dictionary.keys()))
y = np.array(list(sorted_dictionary.values()))
percent = 100.*y/sum(y)
labels_2 = ['{0}  {1:1.2f}%'.format(i,j) for i,j in zip(x, percent)]
colors = cm.Paired(np.arange(len(sizes))/len(sizes))

patches, texts = plt.pie(sorted(sizes, reverse=True), startangle=90, colors=colors, counterclock=False, wedgeprops={'edgecolor': 'w', 'linewidth':1}, radius=1.5)
plt.legend(patches, labels_2, loc="center", prop={'size':8})
plt.title(f'Target classify of PLANet', y=1.1)
center_circle = plt.Circle((0,0), 1.2, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)
plt.savefig('Uw_for_SAR_classify.png', dpi=300, bbox_inches='tight')
plt.close()

# Distribution of activities_count_for_per_target
target_count_df = pd.DataFrame({'count':Uw_target_classify.groupby('Target_chembl_id').size()}).reset_index()
bins=[0, 1, 10, 50, 100, 1000, 3730]
labels=['1', '2-10', '11-50', '51-100', '101-1000', '>1000']
target_count_df['count_group'] = pd.cut(target_count_df['count'], bins=bins, labels=labels)
target_count_df['count_group'].value_counts(sort=False).plot.bar(rot=0)
plt.title('Distribution of activities counts for per target')
plt.xlabel('Number of activities / target')
plt.ylabel('Number of targets')
plt.savefig('distribution_of_activities_count_for_each_target.png', dpi=300, bbox_inches='tight')
plt.close()

# -logAffi
sns.displot(Uw_target_classify, x="-logAffi")
plt.title(f'The distribution of "-logAffi" in PLANet(N={len(Uw_target_classify)})')
plt.savefig('Binding_affinity_distribution_Uw.png', dpi=300, bbox_inches='tight')
plt.close()
