import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
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

def load_dataset(train, valid, test, feature_version, feature_file):
    features = pd.read_csv(feature_file, sep='\t')
    vina_title =['vina_gauss1',
                'vina_gauss2',
                'vina_repulsion',
                'vina_hydrophobic',
                'vina_hydrogen',
                'vina_num_rotors']
    rf_v1_title = features.columns.tolist()[:36]
    rf_v2_title = features.columns.tolist()[36:36+108]

    if feature_version == "V":
        feature_list = vina_title
    elif feature_version == "R1":
        feature_list = rf_v1_title
    elif feature_version == "R2":
        feature_list = rf_v2_title
    elif feature_version == "VR1":
        feature_list = vina_title + rf_v1_title
    elif feature_version == "VR2":
        feature_list = vina_title + rf_v2_title
    elif feature_version == "selected":
        feature_list = ['vina_gauss1','vina_gauss2','vina_hydrophobic','6.6','6.7','6.8']
    elif feature_version == "three_selected":
        feature_list = ['6.6','6.7','6.8']
    elif feature_version == "ten_selected":
        feature_list = ['6.9','6.6','6.15','8.17','8.6','vina_hydrogen','16.16','16.9','7.16','8.15']
    elif feature_version == "nine_selected":
        feature_list = ['6.9','6.6','8.17','8.6','vina_hydrogen','16.16','16.9','7.16','8.15']

    train_code = pd.read_csv(train, sep='\t', header=0, names=['unique_identify', '-logAffi'])
    valid_code = pd.read_csv(valid, sep='\t', header=0, names=['unique_identify', '-logAffi'])
    test_code = pd.read_csv(test, sep='\t', header=0, names=['unique_identify', '-logAffi'])

    train_set = pd.merge(train_code, features, on=['unique_identify','-logAffi'])
    valid_set = pd.merge(valid_code, features, on=['unique_identify','-logAffi'])
    test_set = pd.merge(test_code, features, on=['unique_identify','-logAffi'])

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
            "max_features":list(range(min_tree_feature, max_tree_feature)),
            # "max_depth":[8, 9, 10, 11, 12],
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


def plot_gaussian_kde_scatter_plot(dataset_name, name, res, x, y, path):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)
    ax.plot(x, x)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f'{name} for {dataset_name}')
    ax.text(0.5,0.99,
        res,
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax.transAxes,
        zorder=3)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def rf_grid_search_plot(data, path):
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x="n_estimators", y="R2", hue="max_features")
    ax.set_title("Grid search of RF")
    plt.savefig(path)
    plt.close()