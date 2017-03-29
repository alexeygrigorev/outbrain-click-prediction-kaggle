# Outbrain Click Prediction challenge solution

- The goal of the competition is to predict which ad will be clicked on
- See https://www.kaggle.com/c/outbrain-click-prediction for more details
- This is `ololo`'s part of the 13th place solution to the challenge (team "diaman & ololo")
- The presentation of the solution: http://www.slideshare.net/AlexeyGrigorev/outbrain-click-prediction-71724151
- `diaman`'s solution can be found at https://github.com/dselivanov/kaggle-outbrain


## Overview:

The part of the solution is a combination of 5 models:

- SVM and FTRL on basic features:
  - event features: user id, document id, platform id, day, hour and geo
  - ad features: ad document id, campaign, advertizer id
- XGB and ET on MTV (Mean Target Value) features:
  - all categorical features that previous model used
  - document features like publisher, source, top category, topic and entity
  - interaction between these featuers
  - also, the document similarity features: the cosine between the ad doc and the page with the ad
- FFM with the following features:
  - all categorical features from the above, except document similarity, categories, topics and entities
  - XGB leaves from the previous step (see slide 9 from [this presentation](http://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf) for the description of the idea)
- The models are combined with an XGB model (`rank:pairwise` objective)

To get the 13th positions, models from [diaman](https://www.kaggle.com/dselivanov) should also be added 

## Files description

- `0_prepare_splits.py` splits the training dataset into two folds
- `1_svm_data.py` prepares the data for SVM and FTRL
- `1_train_ftrl.py` and `1_train_svm.py` train models on data from `1_svm_data.py`
- `2_extract_leaked_docs.py` and `2_leak_features.py` extract the leak
- `3_doc_similarity_features.py` calculates TF-IDF similarity between the document user on and the ad document
- `4_categorical_data_join.py` and `4_categorical_data_unwrap_columnwise.py` prepare data for MTV features calculation
- `4_mean_target_value.py` calculates MTV for all features from `categorical_features.txt`
- `5_best_mtv_features_xgb.py` builds an XBG on a small part of data and selects best features to be used on for XGB and ET
- `5_mtv_et.py` trains ET model on MTV features
- `5_mtv_xgb.py` trains XGB model on MTV features and creates leaf featurse to be used in FFM
- `6_1_generate_ffm_data.py` creates the input file to be read by ffmlib
- `6_2_split_ffm_to_subfolds.py` splits each fold into two subfolds (can't use the original folds because the leaf features are not transferable between folds)
- `6_3_run_ffm.sh` runs libffm for training FFM models
- `6_4_put_ffm_subfolds_together.py` puts FFM predictions from each fold/subfold together
- `7_ensemble_data_prep.py` puts all the features and model predictions together for ensembling
- `7_ensemble_xgb.py` traings the second level XGB model on top of all these features

The files should be run in the above order

Diaman's features should be included into `7_ensemble_data_prep.py` - and the rest can stay unchanged.



