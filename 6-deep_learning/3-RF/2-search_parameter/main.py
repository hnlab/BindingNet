from math import gamma
import pandas as pd
import numpy as np
from pathlib import Path
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
from utils import plot_gaussian_kde_scatter_plot
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor


def main(args):
    out_dir = f'{args["output_path"]}/{args["dataset_name"]}/{args["rep_num"]}'
    if not Path(f'{out_dir}').exists():
        Path(f'{out_dir}').mkdir(parents=True)

    ## load features and dataset
    features = pd.read_csv(args["feature_file"], sep='\t')
    vina_title =['vina_gauss1',
                'vina_gauss2',
                'vina_repulsion',
                'vina_hydrophobic',
                'vina_hydrogen',
                'vina_num_rotors']
    rf_v1_title = features.columns.tolist()[:36]
    # rf_v2_title = features.columns.tolist()[36:36+108]
    if args["feature_version"] == "VR1":
        feature_list = vina_title + rf_v1_title
    else:
        print(f'{args["feature_version"] } not implemented now.')

    train_code = pd.read_csv(f'{args["tvt_data_dir"]}/train.csv', sep='\t', header=0, names=['unique_identify', '-logAffi'])
    valid_code = pd.read_csv(f'{args["tvt_data_dir"]}/valid.csv', sep='\t', header=0, names=['unique_identify', '-logAffi'])
    self_test_code = pd.read_csv(f'{args["tvt_data_dir"]}/test.csv', sep='\t', header=0, names=['unique_identify', '-logAffi'])
    core_code = pd.read_csv(args["core_data"], sep='\t', header=0, names=['unique_identify', '-logAffi'])
    core_inter_Uw_code = pd.read_csv(args["core_intersected_Uw"], sep='\t', header=0, names=['unique_identify', '-logAffi'])

    train_set = pd.merge(train_code, features, on=['unique_identify','-logAffi'])
    valid_set = pd.merge(valid_code, features, on=['unique_identify','-logAffi'])
    self_test_set = pd.merge(self_test_code, features, on=['unique_identify','-logAffi'])
    core_set = pd.merge(core_code, features, on=['unique_identify','-logAffi'])
    core_inter_Uw_set = pd.merge(core_inter_Uw_code, features, on=['unique_identify','-logAffi'])

    if args["model"] == "RF":
        model = RandomForestRegressor(n_jobs=6,                            
                min_samples_split=10, 
                n_estimators=args["rf_n_estimator"],
                max_features=args["rf_max_features"],
                verbose=0)

        model.fit(train_set[feature_list], train_set["-logAffi"])
        
    
    elif args["model"]=="XGB":
        model= xgb.XGBRegressor(
            n_estimators = args["xgb_n_estimators"],
            max_depth = args["xgb_max_depth"],
            eta = args["xgb_eta"],
            alpha=args["xgb_alpha"],
            gamma=args["xgb_gamma"],
            n_jobs=6
        )

        model.fit(train_set[feature_list], train_set["-logAffi"])

    elif args["model"] == "LR":
        # model1 = LinearRegression(positive=True, fit_intercept=True).fit(train_set[feature_list], train_set['-logAffi'])
        # model1_valid_r2 = r2_score(model1.predict(valid_set[feature_list]), valid_set['-logAffi'])
        model = LinearRegression(positive=True, fit_intercept=False).fit(train_set[feature_list], train_set['-logAffi'])
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
                    train_set['-logAffi'])
                for i in range(n)))
            

        trained_nets.sort(key=lambda n: n.score(valid_set[feature_list],valid_set['-logAffi']))

        model = ensemble_model(trained_nets[-20:])
    
    elif args["model"] == "NN+":
        model = MLPRegressor((200, 200, 200),
                        batch_size=10,
                        random_state=0,
                        verbose=0,
                        solver='lbfgs')
        
        model.fit(train_set[feature_list], train_set["-logAffi"])


    elif args["model"]=="SGDR":
        model = SGDRegressor(verbose=0, max_iter=100,
                                loss="huber",
                                penalty=args["sgdr_penalty"],
                                alpha=args["sgdr_alpha"],
                                epsilon=args["sgdr_epsilon"],
                                early_stopping=args["sgdr_es"])

        model.fit(train_set[feature_list], train_set["-logAffi"])

    sets = [
    ('Train', train_set['unique_identify'].tolist(), model.predict(train_set[feature_list]), train_set['-logAffi']),
    ('Valid', valid_set['unique_identify'].tolist(), model.predict(valid_set[feature_list]), valid_set['-logAffi']),
    ('Self_test', self_test_set['unique_identify'].tolist(), model.predict(self_test_set[feature_list]), self_test_set['-logAffi']),
    ('Core', core_set['unique_identify'].tolist(), model.predict(core_set[feature_list]), core_set['-logAffi']),
    ('Core_intersected_Uw', core_inter_Uw_set['unique_identify'].tolist(), model.predict(core_inter_Uw_set[feature_list]), core_inter_Uw_set['-logAffi']),
    ]


    results = []
    for name, pdb, pred, target in sets:
        R2_score = r2_score(target, pred)
        Rp = pearsonr(target, pred)[0]
        mae = mean_absolute_error(target, pred)
        result = [name, format(R2_score,'.4f'), format(Rp,'.4f'), format(mae,'.4f')]
        results.append(result)
        res = f'R2:{R2_score:.4f}; Rp:{Rp:.4f}; mae:{mae:.4f}'
        plot_gaussian_kde_scatter_plot(f'{args["dataset_name"]}_{args["rep_num"]}', name, res, pred, target, path=f'{out_dir}/{name}_{args["feature_version"]}_{args["model"]}.png')
        detail_result = pd.DataFrame({'unique_identify':pdb, 'pred_label':pred, 'true_label':target})
        detail_result.to_csv(f'{out_dir}/detail_data_{name}_{args["feature_version"]}_{args["model"]}.csv', index=False)
    results = pd.DataFrame(results, columns=['name','r2','Rp', 'mae'])
    print(results)
    results.to_csv(f'{out_dir}/final_result_{args["feature_version"]}_{args["model"]}.csv', index=False)
    pickle.dump(model, open(f'{out_dir}/best_model_{args["feature_version"]}_{args["model"]}.pkl', "wb"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Protein-Ligand Binding affinity Prediction')
    parser.add_argument("--tvt_data_dir", type=str, default="/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/PDBbind/pdbbind_v2019/minimized/dataset/PDBbind_minimized_rm_core_ids/complex_6A", help="directory of training, validation set")
    parser.add_argument("--core_data", type=str, default="/pubhome/xli02/Downloads/dataset/PDBbind/PDBbind_v2019/CASF_v16_index_dealt.csv", help="CASF_v16 dataset")
    parser.add_argument("--core_intersected_Uw", type=str, default="/pubhome/xli02/project/PLIM/deep_learning/FAST/fast_plim/test_on_core_set/2-core_intersected_Uw/core_intersected_Uw.csv", help="CASF_v16_intersected_Uw dataset")
    parser.add_argument("--rf_max_features", type=int, default=3)
    parser.add_argument("--rf_n_estimator",type=int, default=500)
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
    parser.add_argument("--feature_file", type=str, default="/pubhome/xli02/project/PLIM/deep_learning/RFscore/featured_data/tow_datasets_features_6A.csv", help="feature file")
    parser.add_argument("--dataset_name", type=str, default="PDBbind_minimized_Rm_core", help="dataset name")
    parser.add_argument("--rep_num", type=int, help="repeated number")
    args = parser.parse_args()
    args = parser.parse_args().__dict__
    for k, v in args.items():
        print(f'{k}: {v}')

    main(args)

    