from math import gamma
import pandas as pd
import numpy as np
from sklearn import model_selection
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
from utils import load_dataset,  plot_gaussian_kde_scatter_plot
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor

    

def main(args):
    ## load features and dataset
    feature_list, train_set, valid_set, test_set = load_dataset(args["train_data"], args["valid_data"], args["test_data"], args["feature_version"])
    # train_valid_set = pd.concat([train_set, valid_set])
    # train_valid_set = train_valid_set.sample(frac=1)
    # ## for nn model
    # train_num = int(train_valid_set.shape[0]*0.8)
    # valid_num = train_valid_set.shape[0] - train_num
    # train_set = train_valid_set.head(train_num)
    # valid_set = train_valid_set.tail(valid_num)
    
    if args["model"] == "RF":
        model = RandomForestRegressor(n_jobs=6,                            
                min_samples_split=10, 
                n_estimators=args["rf_n_estimator"],
                max_features=args["rf_max_features"],
                verbose=0)
        

        model.fit(train_set[feature_list], train_set["affinity"])
        
    
    elif args["model"]=="XGB":
        model= xgb.XGBRegressor(
            n_estimators = args["xgb_n_estimators"],
            max_depth = args["xgb_max_depth"],
            eta = args["xgb_eta"],
            alpha=args["xgb_alpha"],
            gamma=args["xgb_gamma"],
            n_jobs=6
        )

        model.fit(train_set[feature_list], train_set["affinity"])

    elif args["model"] == "LR":
        # model1 = LinearRegression(positive=True, fit_intercept=True).fit(train_set[feature_list], train_set['affinity'])
        # model1_valid_r2 = r2_score(model1.predict(valid_set[feature_list]), valid_set['affinity'])
        model = LinearRegression(positive=True, fit_intercept=False).fit(train_set[feature_list], train_set['affinity'])
        # if model1_valid_r2 > model2_valid_r2:
        #     model = model1
        # else:
        #     model = model2
    
    elif args["model"] == "NN":

        n = 1000
        seeds = np.random.randint(123456789, size=n)
        trained_nets = (
            Parallel(n_jobs=5, verbose=0, pre_dispatch='all')(
                delayed(method_caller)(
                    neuralnetwork((args["nn_hidden_size"],),
                                    random_state=seeds[i],
                                    activation='logistic',
                                    solver='lbfgs',
                                    max_iter=10000),
                    'fit',
                    train_set[feature_list],
                    train_set['affinity'])
                for i in range(n)))
            

        trained_nets.sort(key=lambda n: n.score(valid_set[feature_list],valid_set['affinity']))

        model = ensemble_model(trained_nets[-20:])
    
    elif args["model"] == "NN+":
        model = MLPRegressor((200, 200, 200),
                        batch_size=10,
                        random_state=0,
                        verbose=0,
                        solver='lbfgs')
        
        model.fit(train_set[feature_list], train_set["affinity"])


    elif args["model"]=="SGDR":
        model = SGDRegressor(verbose=0, max_iter=100,
                                loss="huber",
                                penalty=args["sgdr_penalty"],
                                alpha=args["sgdr_alpha"],
                                epsilon=args["sgdr_epsilon"],
                                early_stopping=args["sgdr_es"])

        model.fit(train_set[feature_list], train_set["affinity"])

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





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Protein-Ligand Binding affinity Prediction')
    parser.add_argument("--train_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/refine_core/xaa",help="training dataset")
    parser.add_argument("--valid_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/refine_core/xab",help="training dataset")
    parser.add_argument("--test_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/refine_core/core.csv", help="testing dataset")
    parser.add_argument("--rf_max_features", type=int, default=4)
    parser.add_argument("--rf_n_estimator",type=int, default=400)
    parser.add_argument("--xgb_max_depth",type=int)
    parser.add_argument("--xgb_n_estimators",type=int)
    parser.add_argument("--xgb_eta", type=float)
    parser.add_argument("--xgb_gamma", type=int)
    parser.add_argument("--xgb_alpha", type=int)
    parser.add_argument("--nn_hidden_size", type=int, default=5)
    parser.add_argument("--sgdr_penalty", type=str, default="l2")
    parser.add_argument("--sgdr_alpha", type=float, default=0.001)
    parser.add_argument("--sgdr_epsilon", type=float, default=0.1)
    parser.add_argument("--sgdr_es", type=bool, default=False)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--feature_version", type=str, choices=['V','X','C','R1','R2','VR1','VR2','VB','VXC','PLEC','selected','ten_selected','three_selected','VR1_MW'])
    parser.add_argument("--model", type=str, choices=['LR','RF','XGB','NN','SGDR','NN+'])
    args = parser.parse_args()
    args = parser.parse_args().__dict__
    for k, v in args.items():
        print(f'{k}: {v}')

    main(args)

    