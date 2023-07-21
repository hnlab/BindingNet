import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
from scipy.stats import gaussian_kde
import oddt
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from oddt.scoring import scorer, ensemble_model
from oddt.utils import method_caller
from oddt.scoring.models.regressors import neuralnetwork
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.inspection import permutation_importance
import shap
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

features = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/descriptors/refine_general_features.csv")
vina_title =['vina_gauss1_x',
            'vina_gauss2_x',
            'vina_repulsion_x',
            'vina_hydrophobic_x',
            'vina_hydrogen_x',
            'vina_num_rotors']
rf_v1_title = features.columns.tolist()[2:38]


feature_list = vina_title+rf_v1_title

# all_pdb = pd.read_csv("/pubhome/hzhu02/GPSF/dataset/INDEX/split/hmm/jackhmmer/general/general_refine_classified.csv")

# cluster_performance = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/general_3_fold_summary/data_mw_bias/RF_cluster_mw_bias_RCV.csv")
# cluster_performance = cluster_performance.sort_values(by=['Rp'], ascending=False)

# filter_cluster = cluster_performance[cluster_performance['pdb_num']>20].sort_values(by=['Rp'], ascending=False)

# change_feature_list = [item.split("_x")[0] for item in feature_list]

# selected_cluster = ['APC','una_436','una_262','TMEM173','RIP','THDP-binding','Prenyltransf','Sialidase','una_570','Avidin']

for i in range(1,4):
    with open("/pubhome/hzhu02/GPSF/generalization_benchmark/models/RF/general_3_fold/PCV/1/"+str(i)+"/best_model_VR1_RF.pkl", "rb") as f:
            model = joblib.load(f)
    explainer = shap.TreeExplainer(model)

    test_set = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/general_3_fold/PCV/1/"+str(i)+"/train.csv", header=None)
    test_set.columns=['pdb','affinity']
    test_set = pd.merge(test_set, features, on=['pdb','affinity'])
#     test_set = pd.merge(test_set, all_pdb, on=['pdb','affinity'])

    shap_values = explainer.shap_interaction_values(test_set[feature_list])
    np.save("/pubhome/hzhu02/GPSF/generalization_benchmark/models/general_3_fold_summary/RF_feature/pcv_training_shap_"+str(i)+".npy", shap_values)


