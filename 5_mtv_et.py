import pandas as pd
import numpy as np
import feather
import gc

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier


df_train_1 = feather.read_dataframe('tmp/mtv_df_train_1.feather')
features = sorted(set(df_train_1.columns) - {'display_id', 'clicked'})

y_1 = df_train_1.clicked.values
X_1 = df_train_1[features].values

del df_train_1
gc.collect()


df_train_0 = feather.read_dataframe('tmp/mtv_df_train_0.feather')

y_0 = df_train_0.clicked.values
X_0 = df_train_0[features].values

del df_train_0
gc.collect()


# training a model

n_estimators = 100

et_params = dict( 
    criterion='entropy',
    max_depth=40,
    min_samples_split=6,
    min_samples_leaf=6,
    max_features=6, 
    bootstrap=False, 
    n_jobs=-1,
    random_state=1
)


et0 = ExtraTreesClassifier(warm_start=True, **et_params)
et1 = ExtraTreesClassifier(warm_start=True, **et_params)

for n in range(10, n_estimators + 1, 10):
    et0.n_estimators = n
    et0.fit(X_1, y_1)
    pred_0 = et0.predict_proba(X_0)[:, 1]
    s0 = roc_auc_score(y_0, pred_0)

    et1.n_estimators = n
    et1.fit(X_0, y_0)
    pred_1 = et1.predict_proba(X_1)[:, 1]
    s1 = roc_auc_score(y_1, pred_1)
    
    scores = (s0, s1)
    scores_text = ', '.join('%0.5f' % s for s in scores)
    print('%3d, %0.4f, [%s]' % (n, np.mean(scores), scores_text))

print('final scores:', scores)


pred_0 = et0.predict_proba(X_0)[:, 1].astype('float32')
pred_1 = et1.predict_proba(X_1)[:, 1].astype('float32')
del et0, et1


np.save('predictions/et_pred0.npy', pred_0)
np.save('predictions/et_pred1.npy', pred_1)


# training on full dataset

print('full model...')

X = np.concatenate([X_0, X_1])
del X_0, X_1
gc.collect()

y = np.concatenate([y_0, y_1])
del y_0, y_1
gc.collect()


et_full = ExtraTreesClassifier(warm_start=True, **et_params)
et_full.n_estimators = n
et_full.fit(X, y)

del X, y
gc.collect()



# making predictions for test

df_test = feather.read_dataframe('tmp/mtv_df_test.feather')

X_test = df_test[features].values
del df_test


pred_test = et_full.predict_proba(X_test)[:, 1].astype('float32')
np.save('predictions/et_pred_test.npy', pred_test)