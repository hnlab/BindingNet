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
from utils import load_dataset, load_grid_search_model, plot_gaussian_kde_scatter_plot, rf_grid_search_plot
from sklearn.linear_model import SGDRegressor


    

def main(args):
    ## load features and dataset
    feature_list, train_set, valid_set, test_set = load_dataset(args["train_data"], args["valid_data"], args["test_data"], args["feature_version"])
    ## predefine split method
    train_num = train_set.shape[0]
    valid_num = valid_set.shape[0]
    train_indices = np.full((train_num,),-1, dtype=int)
    valid_indices = np.full((valid_num,),0,dtype=int)  
    train_valid_set =pd.concat([train_set, valid_set])

    if args["model"] == "RF":
        model = load_grid_search_model(args["model"], feature_list)
        model.fit(train_valid_set[feature_list], train_valid_set['affinity'])


        for param in model.best_params_.keys():
            print(args["feature_version"], param, model.best_params_[param])

        sets = [
        ('Train', train_set['pdb'].tolist(), model.best_estimator_.predict(train_set[feature_list]), train_set['affinity']),
        ('Valid', valid_set['pdb'].tolist(), model.best_estimator_.predict(valid_set[feature_list]), valid_set['affinity']),
        ('Test', test_set['pdb'].tolist(), model.best_estimator_.predict(test_set[feature_list]), test_set['affinity'])]
        
        

        pickle.dump(model.best_estimator_, open(args["output_path"]+"/RF_"+args["feature_version"]+"_best_model.pkl", "wb"))
    
    
    elif args["model"]=="XGB":
        model = load_grid_search_model(args["model"], feature_list)
        model.fit(train_valid_set[feature_list], train_valid_set['affinity'])

        for param in model.best_params_.keys():
            print(args["feature_version"], param, model.best_params_[param])

        sets = [
        ('Train', train_set['pdb'].tolist(), model.best_estimator_.predict(train_set[feature_list]), train_set['affinity']),
        ('Valid', valid_set['pdb'].tolist(), model.best_estimator_.predict(valid_set[feature_list]), valid_set['affinity']),
        ('Test', test_set['pdb'].tolist(), model.best_estimator_.predict(test_set[feature_list]), test_set['affinity'])]

        pickle.dump(model.best_estimator_, open(args["output_path"]+"/XGB_"+args["feature_version"]+"_best_model.pkl", "wb"))

    elif args["model"] == "LR":
        model1 = LinearRegression(positive=True, fit_intercept=True).fit(train_valid_set[feature_list], train_valid_set['affinity'])
        model1_valid_r2 = r2_score(model1.predict(valid_set[feature_list]), valid_set['affinity'])
        model2 = LinearRegression(positive=True, fit_intercept=False).fit(train_set[feature_list], train_set['affinity'])
        model2_valid_r2 = r2_score(model2.predict(valid_set[feature_list]), valid_set['affinity'])
        if model1_valid_r2 > model2_valid_r2:
            model = model1
        else:
            model = model2

        pickle.dump(model, open(args["output_path"]+"/LR_"+args["feature_version"]+"_best_model.pkl", 'wb'))
        
        sets = [
        ('Train', train_set['pdb'].tolist(), model.predict(train_set[feature_list]), train_set['affinity']),
        ('Valid', valid_set['pdb'].tolist(), model.predict(valid_set[feature_list]), valid_set['affinity']),
        ('Test', test_set['pdb'].tolist(), model.predict(test_set[feature_list]), test_set['affinity'])]
    
    elif args["model"]=="SGDR":
        model = load_grid_search_model(args["model"],feature_list)
        model.fit(train_valid_set[feature_list], train_valid_set['affinity'])
        
        for param in model.best_params_.keys():
            print(args["feature_version"], param, model.best_params_[param])

        sets = [
        ('Train', train_set['pdb'].tolist(), model.best_estimator_.predict(train_set[feature_list]), train_set['affinity']),
        ('Valid', valid_set['pdb'].tolist(), model.best_estimator_.predict(valid_set[feature_list]), valid_set['affinity']),
        ('Test', test_set['pdb'].tolist(), model.best_estimator_.predict(test_set[feature_list]), test_set['affinity'])]
    
    elif args["model"] == "NN":

        hidden_size_param = [5, 10, 15, 20]
        nn_models = []

        for hiddzen_size in hidden_size_param:
            n = 1000
            seeds = np.random.randint(123456789, size=n)
            trained_nets = (
                Parallel(n_jobs=6, verbose=0, pre_dispatch='all')(
                    delayed(method_caller)(
                        neuralnetwork((hiddzen_size,),
                                        random_state=seeds[i],
                                        activation='logistic',
                                        solver='lbfgs',
                                        max_iter=30000),
                        'fit',
                        train_set[feature_list],
                        train_set['affinity'])
                    for i in range(n)))

            trained_nets.sort(key=lambda n: n.score(valid_set[feature_list],valid_set['affinity']))

            middle_model = ensemble_model(trained_nets[-20:])
            nn_models.append(middle_model)

        best_r2 = 0
        for i in range(len(hidden_size_param)):
            r2 = r2_score(nn_models[i].predict(valid_set[feature_list]), valid_set['affinity'])
            if r2 > best_r2:
                best_r2 = r2
                best_hiddzen_size_num = hidden_size_param[i]
                model = nn_models[i]
        
        print(args["feature_version"], "best hiddzen size number is", best_hiddzen_size_num)
        pickle.dump(model, open(args["output_path"]+"/NN_"+args["feature_version"]+"_best_model.pkl", 'wb'))
        
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
        # plot_gaussian_kde_scatter_plot(pred, target, path=args["output_path"]+"/"+name+"_"+args["feature_version"]+"_"+args["model"]+".png")
        detail_result = pd.DataFrame({'pdb':pdb, 'pred_label':pred, 'true_label':target})
        detail_result.to_csv(args["output_path"]+"/detail_data_"+name+"_"+args["feature_version"]+"_"+args["model"]+".csv", index=False)
    results = pd.DataFrame(results, columns=['name','r2','Rp', 'mae'])
    results.to_csv(args["output_path"]+"/final_result_"+args["feature_version"]+"_"+args["model"]+".csv", index=False)






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Protein-Ligand Binding affinity Prediction')
    parser.add_argument("--train_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/general_core/train.csv",help="training dataset")
    parser.add_argument("--valid_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/general_core/valid.csv",help="training dataset")
    parser.add_argument("--test_data", type=str, default="/pubhome/hzhu02/GPSF/generalization_benchmark/datasets/refine_core/core.csv", help="testing dataset")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--feature_version", type=str, choices=['V','X','C','R1','R2','VR1','VR2','VB','VXC','PLEC','selected', 'three_selected','ten_selected','nine_selected','VR1_MW'])
    parser.add_argument("--model", type=str, choices=['LR','RF','XGB','NN','SGDR'])
    args = parser.parse_args()
    args = parser.parse_args().__dict__
    for k, v in args.items():
        print(f'{k}: {v}')

    main(args)