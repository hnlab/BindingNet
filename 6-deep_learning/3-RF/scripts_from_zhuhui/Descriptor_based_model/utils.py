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
from joblib import Parallel, delayed
from oddt.scoring import scorer, ensemble_model
from oddt.utils import method_caller
from oddt.scoring.models.regressors import neuralnetwork
import xgboost as xgb
from sklearn.linear_model import SGDRegressor

def load_dataset(train, valid, test, feature_version):
    features = pd.read_csv("/pubhome/xli02/project/PLIM/deep_learning/RFscore/scripts_from_zhuhui/test_core/core_12A_RFScore_features.csv")
    vina_title =['vina_gauss1',
                'vina_gauss2',
                'vina_repulsion',
                'vina_hydrophobic',
                'vina_hydrogen',
                'vina_num_rotors']
    xscore_title=['x_vwd','x_hb','x_hp','x_hm','x_hs','x_rt'] ## copy from other paper, note: general doesn't have xscore feature
    cyscore_title = ['cy_hydrophobic','cy_vdw','cy_hbond','cy_ent'] ## copy from other paper, note: general doesn't have cyscore feature
    rf_v1_title = features.columns.tolist()[:36]
    rf_v2_title = features.columns.tolist()[36:36+216]
    six_feature = ['vina_gauss1_x',
                'vina_gauss2_x',
                'vina_hydrophobic_x',
                '6.6',
                '6.7',
                '6.8']
    nn_feature_title = features.columns.tolist()[-350:] ## also contain 5 vina features gauss1, gauss2, repulsion, hydrophobic, hydrogen

    if feature_version == "V":
        feature_list = vina_title
    elif feature_version == "X":
        feature_list = xscore_title
    elif feature_version == "C":
        feature_list = cyscore_title
    elif feature_version == "R1":
        feature_list = rf_v1_title
    elif feature_version == "R2":
        feature_list = rf_v2_title
    elif feature_version == "VR1":
        feature_list = vina_title + rf_v1_title
    elif feature_version == "VR2":
        feature_list = vina_title + rf_v2_title
    elif feature_version == "VB":
        feature_list = nn_feature_title
    elif feature_version == "selected":
        feature_list = six_feature
    elif feature_version == "three_selected":
        feature_list = ['6.6','6.7','6.8']
    elif feature_version == "ten_selected":
        feature_list = ['6.9','6.6','6.15','8.17','8.6','vina_hydrogen_x','16.16','16.9','7.16','8.15']
    elif feature_version == "nine_selected":
        feature_list = ['6.9','6.6','8.17','8.6','vina_hydrogen_x','16.16','16.9','7.16','8.15']
    elif feature_version == "PLEC":
        general_features_PLEC = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/PLECScore/extract_PLEC_feature/PLEC_general_feature.csv")
        refine_features_PLEC = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/PLECScore/extract_PLEC_feature/PLEC_feature.csv")
        pdbbind_PLEC = pd.concat([refine_features_PLEC, general_features_PLEC])
        title=pdbbind_PLEC.columns.tolist()
        title.remove("pdb")
        title.remove("affinity")
        feature_list = title
    elif feature_version =="VR1_MW":
        feature_list = vina_title + rf_v1_title + ['mol_weight']
    else: 
        feature_list = vina_title + cyscore_title + xscore_title  ## only for refine

    # all_pdb = pd.read_csv("/pubhome/hzhu02/GPSF/dataset/INDEX/split/hmm/jackhmmer/general/general_refine_classified.csv")
    train_code = pd.read_csv(train, header=None)
    train_code.columns=['pdb', 'affinity']
    train_code = pd.merge(train_code, all_pdb[['pdb','mol_weight']], on=['pdb'])
    valid_code = pd.read_csv(valid, header=None)
    valid_code.columns = ['pdb','affinity']
    valid_code = pd.merge(valid_code, all_pdb[['pdb','mol_weight']], on=['pdb'])
    test_code = pd.read_csv(test, header=None)
    test_code.columns=['pdb', 'affinity']
    test_code = pd.merge(test_code, all_pdb[['pdb','mol_weight']], on=['pdb'])

    if feature_version != "PLEC":
        try:
            train_set = pd.merge(train_code, features, on=['pdb','affinity'])
            valid_set = pd.merge(valid_code, features, on=['pdb','affinity'])
            test_set = pd.merge(test_code, features, on=['pdb','affinity'])
        except:
            train_set = pd.merge(train_code, features, on=['pdb'])
            train_set = train_set.rename(columns={"affinity_y":"affinity"})
            valid_set = pd.merge(valid_code, features, on=['pdb'])
            valid_set = valid_set.rename(columns={"affinity_y":"affinity"})
            test_set = pd.merge(test_code, features, on=['pdb'])
            test_set = test_set.rename(columns={"affinity_y":"affinity"})

    else:
        try:
            train_set = pd.merge(train_code, pdbbind_PLEC, on=['pdb','affinity'])
            valid_set = pd.merge(valid_code, pdbbind_PLEC, on=['pdb','affinity'])
            test_set = pd.merge(test_code, pdbbind_PLEC, on=['pdb','affinity'])
        except:
            train_set = pd.merge(train_code, pdbbind_PLEC, on=['pdb'])
            train_set = train_set.rename(columns={"affinity_y":"affinity"})
            valid_set = pd.merge(valid_code, pdbbind_PLEC, on=['pdb'])
            valid_set = valid_set.rename(columns={"affinity_y":"affinity"})
            test_set = pd.merge(test_code, pdbbind_PLEC, on=['pdb'])
            test_set = test_set.rename(columns={"affinity_y":"affinity"})


    return feature_list, train_set, valid_set, test_set
    

def load_grid_search_model(model_version, feature_list):
    if model_version == "RF" :
        model = RandomForestRegressor(n_jobs=6,                            
                            min_samples_split=10, 
                            verbose=0)
        
        
        if len(feature_list) < 50:
            min_tree_feature = 3
            max_tree_feature = min(9, len(feature_list)+1)
        
        else:
            min_tree_feature = 6
            max_tree_feature = 15

        

        model_search = GridSearchCV(model, {
            "n_estimators":[100,200,300,400,500],
            "max_features":list(range(min_tree_feature, max_tree_feature))
        }, verbose=0, scoring="r2",cv=5)

    if model_version == "XGB":
        model = xgb.XGBRegressor()
        model_search = GridSearchCV(model, {'max_depth': [2, 4, 6],
                    'n_estimators': [100, 200, 300, 400, 500], 
                    "eta":[0.1, 0.01, 0.001], 
                    "gamma":[0,1,2], 
                    "alpha":[0,1] }, verbose=1, n_jobs=6, cv=5)

    if model_version == "SGDR" :
        model = SGDRegressor(verbose=0, max_iter=100)

        model_search = GridSearchCV(model, {
                    'loss':['huber'],
                    'penalty':['l2','l1','elasticnet'],
                    'alpha':[1e-3, 1e-4, 1e-5],
                    'epsilon':[0.1, 0.01],
                    'early_stopping':[True, False]}, verbose=0, scoring="r2")


    
    return model_search


def plot_gaussian_kde_scatter_plot(x, y, path):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)
    ax.plot(x, x)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.savefig(path)
    plt.show()

def rf_grid_search_plot(data, path):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x="n_estimators", y="R2", hue="max_features")
    ax.set_title("Grid search of RF")
    plt.savefig(path)