from oddt.scoring import descriptors
from oddt.scoring.descriptors.binana import binana_descriptor
import pandas as pd
import numpy as np
import oddt
from oddt import toolkit
# from oddt.scoring.descriptors import close_contacts_descriptor, oddt_vina_descriptor




def extract_nn_feature(ligand, protein):
    descriptors_nn = binana_descriptor()
    feature = descriptors_nn.build([ligand], protein)
    return descriptors_nn.titles, feature 





pdbbind = pd.read_csv("/pubhome/hzhu02/GPSF/dataset/INDEX/split/hmm/jackhmmer/general/general_affinity_mw.csv", sep=",")
# pdbbind = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/RFScore/test.csv", header=None, sep=",")
# pdbbind.columns=['pdb','affinity']


for i in range(pdbbind.shape[0]):
    pdb = pdbbind.iloc[i,]['pdb']
    print(pdb)
    protein = next(oddt.toolkits.ob.readfile('pdb', '/pubhome/hzhu02/GPSF/dataset/pdbbind_v2020/v2020-other-PL/'+pdb+'/'+pdb+'_protein.pdb'))
    ligand = next(oddt.toolkits.ob.readfile('sdf', '/pubhome/hzhu02/GPSF/dataset/pdbbind_v2020/v2020-other-PL/'+pdb+'/'+pdb+'_ligand.sdf'))
    if i == 0:
        title, feature = extract_nn_feature(ligand, protein)
    else:
        _, feature_nn = extract_nn_feature(ligand, protein)
        feature = np.vstack((feature, feature_nn))




data = pd.DataFrame(feature, columns=title)
data['pdb'] = pdbbind['pdb'].tolist()
data['affinity'] = pdbbind['affinity'].tolist()

data.to_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/NNScore/extract_nn_feature/nnscore_general_feature.csv", index=False)
    




