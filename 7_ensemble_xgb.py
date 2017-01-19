import pandas as pd
import numpy as np
import xgboost as xgb
import feather
import gc

# prapare the data matrices


df_train_0 = feather.read_dataframe('tmp/df_train_0_ensemble.feather')

ignore = {'display_id', 'ad_id', 'clicked', 'fold'}
columns = sorted(set(df_train_0.columns) - ignore)

group0_sizes = df_train_0.display_id.value_counts(sort=False)
group0_sizes.sort_index(inplace=1)
group0_sizes = group0_sizes.values.astype('uint8')

y_0 = df_train_0.clicked.values
X_0 = df_train_0[columns].values
del df_train_0
gc.collect()

dfold0 = xgb.DMatrix(X_0, y_0, feature_names=columns)
dfold0.set_group(group0_sizes)

del X_0, y_0
gc.collect()



df_train_1 = feather.read_dataframe('tmp/df_train_1_ensemble.feather')

group1_sizes = df_train_1.display_id.value_counts(sort=False)
group1_sizes.sort_index(inplace=1)
group1_sizes = group1_sizes.values.astype('uint8')

y_1 = df_train_1.clicked.values
X_1 = df_train_1[columns].values
del df_train_1
gc.collect()

dfold1 = xgb.DMatrix(X_1, y_1, feature_names=columns)
dfold1.set_group(group1_sizes)

del X_1, y_1
gc.collect()

watchlist = [(dfold0, 'train'), (dfold1, 'val')]


# train the model

n_estimators = 1000

xgb_pars = {
    'eta': 0.15,
    'gamma': 0.0,
    'max_depth': 8,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0,
    'tree_method': 'approx',
    'objective': 'rank:pairwise',
    'eval_metric': 'map@12',
    'nthread': 12,
    'seed': 42,
    'silent': 1
}


# train the model

model = xgb.train(xgb_pars, dfold0, num_boost_round=n_estimators, 
                  verbose_eval=1, evals=watchlist)

del dfold0, dfold1, watchlist
gc.collect()


# test predict

df_test = feather.read_dataframe('tmp/df_test_ensemble.feather')

group_test_sizes = df_test.display_id.value_counts(sort=False)
group_test_sizes.sort_index(inplace=1)
group_test_sizes = group_test_sizes.values.astype('uint8')

X_test = df_test[columns].values
df_test = df_test[['display_id', 'ad_id']].copy()

dtest = xgb.DMatrix(X_test, feature_names=columns)
dtest.set_group(group_test_sizes)
del X_test


test_pred = model.predict(dtest)
df_test['pred'] = test_pred


feather.write_dataframe(df_test, 'final_submission.feather')

# now run `Rscript submission.R final_submission.feather xgb_submission.csv`