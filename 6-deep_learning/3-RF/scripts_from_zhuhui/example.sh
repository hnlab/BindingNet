## 1.extract rfscore feature
python ./extract_feature/extract_rf_feature.py
## 2.change the feature file in utils.py line 23
## 3.search best parameters
python ./Descriptor_based_model/search_best_param.py --model RF --feature_version VR1 
## 4. train RFScore
python ./Descriptor_based_model/main.py --model RF --feature_version VR1 --rf_max_features 8 --rf_n_estimator 500
