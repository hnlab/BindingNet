from oddt.scoring import descriptors
from oddt.scoring.descriptors.binana import binana_descriptor
import pandas as pd
import numpy as np
import oddt
from oddt import toolkit
from oddt.scoring.descriptors import universal_descriptor
from oddt.fingerprints import PLEC, MAX_HASH_VALUE
from functools import partial
from scipy.sparse import coo_matrix, vstack




def extract_PLEC_feature(ligand, protein):
    plec_func = partial(PLEC,
                    depth_ligand=1,
                    depth_protein=5, ## The depth of ECFP environments generated on the protein side of interaction. By default 6 (0 to 5) environments are generated.                                       
                    size=65536, ##The final size of a folded PLEC fingerprint. 
                    count_bits=True,
                    sparse=True,
                    ignore_hoh=True)
    descriptors = universal_descriptor(plec_func, shape=65536, sparse=True)
    feature = descriptors.build([ligand], protein)
    return descriptors.titles, feature 





pdbbind = pd.read_csv("/pubhome/hzhu02/GPSF/dataset/INDEX/split/hmm/jackhmmer/general/general_affinity_mw.csv", sep=",")
# pdbbind = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/RFScore/test.csv", header=None, sep=",")
# pdbbind.columns=['pdb','affinity']


for i in range(pdbbind.shape[0]):
    pdb = pdbbind.iloc[i,]['pdb']
    print(pdb)
    protein = next(oddt.toolkits.ob.readfile('pdb', '/pubhome/hzhu02/GPSF/dataset/pdbbind_v2020/v2020-other-PL/'+pdb+'/'+pdb+'_protein.pdb'))
    ligand = next(oddt.toolkits.ob.readfile('sdf', '/pubhome/hzhu02/GPSF/dataset/pdbbind_v2020/v2020-other-PL/'+pdb+'/'+pdb+'_ligand.sdf'))
    if i == 0:
        title, feature = extract_PLEC_feature(ligand, protein)
    else:
        _, feature_PLEC = extract_PLEC_feature(ligand, protein)
        feature = vstack([feature, feature_PLEC])




data = pd.DataFrame(feature.toarray())
data['pdb'] = pdbbind['pdb'].tolist()
data['affinity'] = pdbbind['affinity'].tolist()

data.to_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/PLECScore/extract_PLEC_feature/PLEC_general_feature.csv", index=False)
    




