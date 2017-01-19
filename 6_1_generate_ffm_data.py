import pandas as pd
import numpy as np

from tqdm import tqdm
import feather
import gc


D = 2 ** 20

# display features
USER = '0'
ON_DOC = '1'
PLATFORM = '2'

# ads features
AD = '3'
AD_DOC = '4'
CAMPAIGN = '5'
ADVERTISER = '6'

# document features
ON_SRC = '7'
ON_PUBLISHER = '8'

AD_SRC = '9'
AD_PUBLISHER = '10' 


def hash_element(el):
    h = hash(el) % D
    if h < 0:
        h = h + D
    return str(h)


# reading the events features

df_events = pd.read_csv("../data/events.csv", usecols=['uuid', 'document_id', 'platform'])

user_str = USER + ':' + df_events.uuid.apply(hash_element) + ':1'
doc_str = ON_DOC + ':' + df_events.document_id.apply(hash_element) + ':1'
platforms = PLATFORM + ':' + df_events.platform.astype('str') + ':1'

df_events_processed = pd.DataFrame()
df_events_processed['display_str'] = user_str + ' ' + doc_str + ' ' + platforms
df_events_processed['document_id'] = df_events.document_id

del df_events, user_str, doc_str, platforms


# reading the ads features

df_ads = pd.read_csv("../data/promoted_content.csv")
ad_to_doc = dict(zip(df_ads.ad_id, df_ads.document_id))

ad_str = AD + ':' + df_ads.ad_id.astype(str) + ':1 ' + \
         AD_DOC + ':' + df_ads.document_id.apply(hash_element) + ':1 ' + \
         CAMPAIGN + ':' + df_ads.campaign_id.astype(str) + ':1 ' + \
         ADVERTISER + ':' + df_ads.advertiser_id.astype(str) + ':1'

ad_str_dict = dict(zip(df_ads.ad_id, ad_str))

del ad_str, df_ads


# reading the document meta features - others aren't included

df_doc_meta = pd.read_csv('../data/documents_meta.csv')

df_doc_meta.source_id.fillna(0, inplace=1)
df_doc_meta.source_id = df_doc_meta.source_id.astype('int32')
df_doc_meta.publisher_id.fillna(0, inplace=1)
df_doc_meta.publisher_id = df_doc_meta.publisher_id.astype('int32')
del df_doc_meta['publish_time']

meta_src = df_doc_meta.source_id.astype('str') + ':1 '
meta_src_dict = dict(zip(df_doc_meta.document_id, meta_src))

meta_pub = df_doc_meta.publisher_id.astype('str') + ':1'
meta_pub_dict = dict(zip(df_doc_meta.document_id, meta_pub))

del df_doc_meta, meta_src, meta_pub

# generating the ffm data 

leaves_start = 11

def ffm_feature_string(display_id, ad_id, leaves, label=None):
    ad_doc_id = ad_to_doc[ad_id]

    ad_features = ad_str_dict[ad_id] #
    
    disp_row = df_events_processed.iloc[display_id - 1]
    on_doc_id = disp_row.document_id
    disp_features = disp_row.display_str #

    on_src = ON_SRC + ':' + meta_src_dict[on_doc_id]
    on_pub = ON_PUBLISHER + ':' + meta_pub_dict[on_doc_id]

    ad_src = AD_SRC + ':' + meta_src_dict[ad_doc_id]
    ad_pub = AD_PUBLISHER + ':' + meta_pub_dict[ad_doc_id]
    
    leaves_features = []

    for i, leaf in enumerate(leaves):
        leaves_features.append('%d:%d:1' % (leaves_start + i, leaf))

    leaves_features = ' '.join(leaves_features)

    result = disp_features + ' ' + ad_features + ' ' + \
             on_src + ' ' + on_pub + ' ' + \
             ad_src + ' ' + ad_pub + ' ' + \
             leaves_features

    if label is None:
        return '0 ' + result
    else:
        return str(label) + ' ' + result


# generating the data for train

df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')

leaves_0 = np.load('tmp/xgb_model_0_leaves.npy')
leaves_1 = np.load('tmp/xgb_model_1_leaves.npy')


f_0 = open('ffm/ffm_xgb_0.txt', 'w')
f_1 = open('ffm/ffm_xgb_1.txt', 'w')
cnt_0 = 0
cnt_1 = 0

for row in tqdm(df_all.itertuples()):
    display_id = row.display_id
    ad_id = row.ad_id
    fold = row.fold
    label = row.clicked

    if fold == 0:
        row = ffm_feature_string(display_id, ad_id, leaves_0[cnt_0], label)
        f_0.write(row + '\n')
        cnt_0 = cnt_0 + 1
    else:
        row = ffm_feature_string(display_id, ad_id, leaves_1[cnt_1], label)
        f_1.write(row + '\n')
        cnt_1 = cnt_1 + 1

f_0.close()
f_1.close()


del df_all, leaves_0, leaves_1
gc.collect()


# generating the data for test

df_test = feather.read_dataframe('tmp/clicks_test.feather')

leaves_0 = np.load('tmp/xgb_model_0_test_leaves.npy')
leaves_1 = np.load('tmp/xgb_model_1_test_leaves.npy')

f_0 = open('ffm/ffm_xgb_test_0.txt', 'w')
f_1 = open('ffm/ffm_xgb_test_1.txt', 'w')

cnt = 0

for row in tqdm(df_test.itertuples()):
    display_id = row.display_id
    ad_id = row.ad_id

    row = ffm_feature_string(display_id, ad_id, leaves_0[cnt])
    f_0.write(row + '\n')

    row = ffm_feature_string(display_id, ad_id, leaves_1[cnt])
    f_1.write(row + '\n')
    
    cnt = cnt + 1

f_0.close()
f_1.close()