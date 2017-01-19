# coding: utf-8

from time import time

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score


# building the data for train

df_all = pd.read_csv('tmp/svm_features_train.csv')

text_vec = HashingVectorizer(dtype=np.uint8, n_features=10000000, norm=None, 
                             lowercase=False, binary=True, token_pattern='\\S+', 
                             non_negative=True)

t0 = time()
X = text_vec.transform(df_all.ad_display_str)

print('building the train matrix took %.4fm' % (time() - t0) / 60)


fold = df_all.fold.values

X_0 = X[fold == 0]
X_1 = X[fold == 1]

y = df_all.clicked.values
y_0 = y[fold == 0]
y_1 = y[fold == 0]


# fitting the model for fold 1

C = 0.1

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, C=C, random_state=1)
svm.fit(X_0, y_0)

y_pred = svm.decision_function(X_1)
auc = roc_auc_score(y_1, y_pred)

np.save('predictions/svm_1_preds.npy', y_pred)

print('C=%s, took %.3fs, auc=%.3f' % (C, time() - t0, auc))


# fitting the model for fold 0

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, C=C, random_state=1)
svm.fit(X_1, y_1)

y_pred = svm.decision_function(X_0)
auc = roc_auc_score(y_0, y_pred)

np.save('predictions/svm_0_preds.npy', y_pred)

print('C=%s, took %.3fs, auc=%.3f' % (C, time() - t0, auc))


# predictions for test

df_test = pd.read_csv('tmp/svm_features_test.csv')

t0 = time() 
X_test = text_vec.transform(df_test.ad_display_str)

print('building the test matrix took %.4fm' % (time() - t0) / 60)

pred_0 = model_0.decision_function(X_test)
pred_1 = model_1.decision_function(X_test)
pred_final = (pred_0 + pred_1) / 2

np.save('predictions/svm_test_preds.npy', pred_final)