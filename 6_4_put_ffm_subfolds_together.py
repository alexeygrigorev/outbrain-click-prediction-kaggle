import feather
import numpy as np


df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')

df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)
del df_train_0['fold'], df_train_1['fold'], df_all

df_train_0['fold'] = np.load('tmp/fold_0_split.npy')
df_train_1['fold'] = np.load('tmp/fold_1_split.npy')


#  predictions of two subfolds of fold 0

pred_0_0 = pd.read_csv('ffm/pred_0_0.txt', header=None, dtype='float32')
pred_0_0 = pred_0_0[0]

pred_0_1 = pd.read_csv('ffm/pred_0_1.txt', header=None, dtype='float32')
pred_0_1 = pred_0_1[0]


df_train_0.loc[df_train_0.fold == 0, 'ffm_xgb'] = pred_0_0.values
df_train_0.loc[df_train_0.fold == 1, 'ffm_xgb'] = pred_0_1.values
ffm_xgb_0 = df_train_0.ffm_xgb.astype('float32')

np.save('predictions/ffm_0.npy', ffm_xgb_0.values)


#  predictions of two subfolds of fold 1

pred_1_0 = pd.read_csv('ffm/pred_1_0.txt', header=None, dtype='float32')
pred_1_0 = pred_1_0[0]

pred_1_1 = pd.read_csv('ffm/pred_1_1.txt', header=None, dtype='float32')
pred_1_1 = pred_1_1[0]


df_train_1.loc[df_train_1.fold == 0, 'ffm_xgb'] = pred_1_0.values
df_train_1.loc[df_train_1.fold == 1, 'ffm_xgb'] = pred_1_1.values
ffm_xgb_1 = df_train_1.ffm_xgb.astype('float32')

np.save('predictions/ffm_1.npy', ffm_xgb_1.values)


# test predictions

pred_test_0 = pd.read_csv('ffm/pred_test_0.txt', header=None, dtype='float32')
pred_test_1 = pd.read_csv('ffm/pred_test_1.txt', header=None, dtype='float32')

pred_test = (pred_test_0[0] + pred_test_1[0]) / 2
pred_test = pred_test.astype('float32')

np.save('predictions/ffm_test.npy', pred_test.values)