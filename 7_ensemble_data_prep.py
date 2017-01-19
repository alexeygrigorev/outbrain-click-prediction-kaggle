import pandas as pd
import numpy as np
import feather

from tqdm import tqdm


df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')
df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)
del df_train_0['fold'], df_train_1['fold'], df_all

df_test = feather.read_dataframe('tmp/clicks_test.feather')


# read svm predictions

df_train_0['svm'] = np.load('predictions/svm_0_preds.npy')
df_train_0['svm'] = df_train_0['svm'].astype('float32')

df_train_1['svm'] = np.load('predictions/svm_1_preds.npy')
df_train_1['svm'] = df_train_1['svm'].astype('float32')

df_test['svm'] = np.load('predictions/svm_test_preds.npy')
df_test['svm'] = df_test['svm'].astype('float32')


# read ftrl predictions

ftrl_0 = pd.read_csv('predictions/ftrl_pred_0.txt')
df_train_0['ftrl'] = ftrl_0.y_pred.astype('float32')

ftrl_1 = pd.read_csv('predictions/ftrl_pred_1.txt')
df_train_1['ftrl'] = ftrl_1.y_pred.astype('float32')

ftrl_test = pd.read_csv('predictions/ftrl_pred_test.txt')
df_test['ftrl'] = ftrl_test.y_pred.astype('float32')


# read xgb predictions

df_train_0['xgb_mtv'] = np.load('predictions/xgb_mtv_pred0.npy')
df_train_1['xgb_mtv'] = np.load('predictions/xgb_mtv_pred1.npy')
df_test['xgb_mtv'] = np.load('predictions/xgb_mtv_pred_test.npy')


# read et predictions

df_train_0['et_mtv'] = np.load('predictions/et_pred0.npy')
df_train_1['et_mtv'] = np.load('predictions/et_pred1.npy')
df_test['et_mtv'] = np.load('predictions/et_pred_test.npy')


# read ffm predictions

df_train_0['ffm'] = np.load('predictions/ffm_0.npy')
df_train_1['ffm'] = np.load('predictions/ffm_1.npy')
df_test['ffm'] = np.load('predictions/ffm_test.npy')


# read the leak features

df_train_0['leak'] = np.load('features/leak_0.npy')
df_train_0['leak'] = df_train_0['leak'].astype('uint8')

df_train_1['leak'] = np.load('features/leak_1.npy')
df_train_1['leak'] = df_train_1['leak'].astype('uint8')

df_test['leak'] = np.load('features/leak_test.npy')
df_test['leak'] = df_test['leak'].astype('uint8')


df_train_0['doc_known_views'] = np.load('features/doc_known_views_0.npy')
df_train_0['doc_known_views'] = df_train_0['leak'].astype('uint32')

df_train_1['doc_known_views'] = np.load('features/doc_known_views_1.npy')
df_train_1['doc_known_views'] = df_train_1['leak'].astype('uint32')

df_test['doc_known_views'] = np.load('features/doc_known_views_test.npy')
df_test['doc_known_views'] = df_test['leak'].astype('uint32')


# rank features

cols_to_rank = ['svm', 'ftrl', 'xgb_mtv', 'et_mtv', 'ffm']


for f in tqdm(cols_to_rank):
    for df in [df_train_0, df_train_1, df_test]:
        df['%s_rank' % f] = df.groupby('display_id')[f].rank(method='dense', ascending=0)
        df['%s_rank' % f] = df['%s_rank' % f].astype('uint8')


# some mean target value features

mtv_features = ['ad_document_id_on_doc_publisher_id',
                'ad_doc_source_id_on_doc_publisher_id',
                'ad_document_id_on_doc_source_id']

for f in mtv_features:
    df_train_0[f] = np.load('features/mte/%s_pred_0.npy' % f)
    df_train_0['%s_rank' % f] = np.load('features/mte/%s_pred_rank_0.npy' % f)

    df_train_1[f] = np.load('features/mte/%s_pred_1.npy' % f)
    df_train_1['%s_rank' % f] = np.load('features/mte/%s_pred_rank_1.npy' % f)

    df_test[f] = np.load('features/mte/%s_pred_test.npy' % f)
    df_test['%s_rank' % f] = np.load('features/mte/%s_pred_rank_test.npy' % f)


# now save everything

feather.write_dataframe(df_train_0, 'df_train_0_ensemble.feather')
feather.write_dataframe(df_train_1, 'df_train_1_ensemble.feather')
feather.write_dataframe(df_test, 'df_test_ensemble.feather')
