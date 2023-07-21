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
from sklearn.neural_network import MLPRegressor
from utils import load_dataset,  plot_gaussian_kde_scatter_plot
from sklearn.linear_model import SGDRegressor




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Protein-Ligand Binding affinity Prediction')
    parser.add_argument("--train_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/refine_core/xaa",help="training dataset")
    parser.add_argument("--valid_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/refine_core/xab",help="training dataset")
    parser.add_argument("--test_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/refine_core/core.csv", help="testing dataset")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_iter", type=int, default=100)
    # parser.add_argument("--specific_valid", type=bool, default=False)
    parser.add_argument("--feature_version", type=str, default="PLEC", choices=['V','X','C','R1','R2','VR1','VR2','VB','VXC','PLEC'])
    parser.add_argument("--model", type=str, choices=['LR','RF','XGB','NN'])
    args = parser.parse_args()
    args = parser.parse_args().__dict__
    for k, v in args.items():
        print(f'{k}: {v}')

    features = pd.read_csv("/pubhome/hzhu02/GPSF/generalization_benchmark/models/PLECScore/extract_PLEC_feature/PLEC_feature.csv")
    title=features.columns.tolist()
    title.remove("pdb")
    title.remove("affinity")
    feature_list = title


    train_code = pd.read_csv(args["train_data"], header=None)
    train_code.columns=['pdb', 'affinity']
    valid_code = pd.read_csv(args["valid_data"], header=None)
    valid_code.columns = ['pdb','affinity']
    test_code = pd.read_csv(args["test_data"], header=None)
    test_code.columns=['pdb', 'affinity']

    train_set = pd.merge(train_code, features, on=['pdb','affinity'])
    valid_set = pd.merge(valid_code, features, on=['pdb','affinity'])
    test_set = pd.merge(test_code, features, on=['pdb','affinity'])

    train_valid_set =pd.concat([train_set, valid_set])
    train_valid_set = train_valid_set.sample(frac=1)

    if args["model"]=="RF":

        model = RandomForestRegressor(n_estimators=500,
                                        n_jobs=4,
                                        verbose=0,
                                        random_state=0)


        model.fit(train_valid_set[feature_list], train_valid_set['affinity'])

        sets = [
        ('Train', train_set['pdb'].tolist(), model.predict(train_set[feature_list]), train_set['affinity']),
        ('Valid', valid_set['pdb'].tolist(), model.predict(valid_set[feature_list]), valid_set['affinity']),
        ('Test', test_set['pdb'].tolist(), model.predict(test_set[feature_list]), test_set['affinity'])]

    elif args["model"]=="NN":
        model = MLPRegressor((200, 200, 200),
                                 batch_size=10,
                                 random_state=0,
                                 verbose=0,
                                 solver='lbfgs')

        model.fit(train_set[feature_list], train_set['affinity'])

        sets = [
        ('Train', train_set['pdb'].tolist(), model.predict(train_set[feature_list]), train_set['affinity']),
        ('Valid', valid_set['pdb'].tolist(), model.predict(valid_set[feature_list]), valid_set['affinity']),
        ('Test', test_set['pdb'].tolist(), model.predict(test_set[feature_list]), test_set['affinity'])]

    elif args["model"]=="LR":
        kwargs = {'fit_intercept': False,
            'loss': 'huber',
            'penalty': 'elasticnet',
            'random_state': 0,
            'verbose': 0,
            'alpha': 1e-4,
            'epsilon': 1e-1,
            'max_iter':100,
            'early_stopping':False
            }
        model = SGDRegressor(**kwargs)
        model.fit(train_valid_set[feature_list], train_valid_set['affinity'])

        sets = [
        ('Train', train_set['pdb'].tolist(), model.predict(train_set[feature_list]), train_set['affinity']),
        ('Valid', valid_set['pdb'].tolist(), model.predict(valid_set[feature_list]), valid_set['affinity']),
        ('Test', test_set['pdb'].tolist(), model.predict(test_set[feature_list]), test_set['affinity'])]



    
    results = []
    for name, pdb, pred, target in sets:
        R2_score = r2_score(target, pred)
        Rp = pearsonr(target, pred)[0]
        mae = mean_absolute_error(target, pred)
        result = [name, format(R2_score,'.4f'), format(Rp,'.4f'), format(mae,'.4f')]
        results.append(result)
        plot_gaussian_kde_scatter_plot(pred, target, path=args["output_path"]+"/"+name+"_"+args["feature_version"]+"_"+args["model"]+".png")
        detail_result = pd.DataFrame({'pdb':pdb, 'pred_label':pred, 'true_label':target})
        detail_result.to_csv(args["output_path"]+"/detail_data_"+name+"_"+args["feature_version"]+"_"+args["model"]+".csv", index=False)
    
    
    results = pd.DataFrame(results, columns=['name','r2','Rp', 'mae'])
    results.to_csv(args["output_path"]+"/final_result_"+args["feature_version"]+"_"+args["model"]+".csv", index=False)
    pickle.dump(model, open(args["output_path"]+"/best_model_"+args["feature_version"]+"_"+args["model"]+".pkl", "wb"))

