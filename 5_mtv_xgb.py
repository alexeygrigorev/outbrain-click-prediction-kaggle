import pandas as pd
import numpy as np
import xgboost as xgb
import feather
import gc


df_train_1 = feather.read_dataframe('tmp/mtv_df_train_1.feather')
features = sorted(set(df_train_1.columns) - {'display_id', 'clicked'})

y_1 = df_train_1.clicked.values
X_1 = df_train_1[features].values
del df_train_1

dfold1 = xgb.DMatrix(X_1, y_1, feature_names=features)
del X_1, y_1
gc.collect()


df_train_0 = feather.read_dataframe('tmp/mtv_df_train_0.feather')

y_0 = df_train_0.clicked.values
X_0 = df_train_0[features].values
del df_train_0
gc.collect()

dfold0 = xgb.DMatrix(X_0, y_0, feature_names=features)
del X_0, y_0
gc.collect()



# training a model

n_estimators = 100

xgb_pars = {
    'eta': 0.2,
    'gamma': 0.5,
    'max_depth': 6,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 1,
    'colsample_bytree': 0.5,
    'colsample_bylevel': 0.5,
    'lambda': 1,
    'alpha': 0,
    'tree_method': 'approx',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 20,
    'seed': 42,
    'silent': 1
}


print('training model on fold 0...')

watchlist = [(dfold0, 'train'), (dfold1, 'val')]
model_fold1 = xgb.train(xgb_pars, dfold0, num_boost_round=n_estimators, 
                        verbose_eval=1, evals=watchlist)

print('training model on fold 1...')

watchlist = [(dfold1, 'train'), (dfold0, 'val')]
model_fold0 = xgb.train(xgb_pars, dfold1, num_boost_round=n_estimators, 
                        verbose_eval=1, evals=watchlist)


pred0 = model_fold0.predict(dfold0)
pred1 = model_fold1.predict(dfold1)

np.save('predictions/xgb_mtv_pred0.npy', pred0)
np.save('predictions/xgb_mtv_pred1.npy', pred1)


# saving the training leaves

leaves0 = model_0.predict(dfold0, pred_leaf=True).astype('uint8')

np.save('tmp/xgb_model_0_leaves.npy', leaves0)
del leaves0
gc.collect()


leaves1 = model_1.predict(dfold1, pred_leaf=True).astype('uint8')

np.save('tmp/xgb_model_1_leaves.npy', leaves1)
del leaves1
gc.collect()



# making prediction for test and getting the leaves

df_test = feather.read_dataframe('tmp/mtv_df_test.feather')


X_test = df_test[features].values
del df_test
gc.collect()

dtest = xgb.DMatrix(X_test, feature_names=features)
del X_test
gc.collect()


pred0_test = model_0.predict(dtest)
pred1_test = model_1.predict(dtest)
pred_test = (pred0_test + pred1_test) / 2

np.save('predictions/xgb_mtv_pred_test.npy', pred_test)


# predicting leaves for test

leaves0_test = model_0.predict(dtest, pred_leaf=True).astype('uint8')
np.save('tmp/xgb_model_0_test_leaves.npy', leaves0_test)

del leaves0_test
gc.collect()

leaves1_test = model_1.predict(dtest, pred_leaf=True).astype('uint8')
np.save('tmp/xgb_model_1_test_leaves.npy', leaves1_test)

del leaves1_test
gc.collect()