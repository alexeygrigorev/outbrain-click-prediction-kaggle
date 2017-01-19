# coding: utf-8

import feather
import sys

import csv
csv.field_size_limit(sys.maxsize)

import pandas as pd
import numpy as np

# reading the leaked documents

docs_size = {}
leak_uuid_dict = {}

with open("tmp/leaked_docs.csv") as f:
    reader = csv.DictReader(f)
    leak_uuid_dict = {}

    for row in reader:
        doc_id = int(row['document_id'])
        uuids = row['uuids'].split(' ')
        leak_uuid_dict[doc_id] = set(uuids)
        docs_size[doc_id] = len(uuids)


# 

df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')
df_test = feather.read_dataframe('tmp/clicks_test.feather')


# getting user ids and document ids

df_events = pd.read_csv('../data/events.csv', usecols=['uuid'])
df_ads = pd.read_csv('../data/promoted_content.csv', 
                     usecols=['ad_id', 'document_id'])

# joining doc_id and ad_id

ad_to_idx = dict(zip(df_ads.ad_id, df_ads.index))

ad_idx = df_all.ad_id.apply(ad_to_idx.get)
ad_document_id = df_ads.document_id.iloc[ad_idx].reset_index(drop=1)
df_all['ad_document_id'] = ad_document_id

ad_idx = df_test.ad_id.apply(ad_to_idx.get)
ad_document_id = df_ads.document_id.iloc[ad_idx].reset_index(drop=1)
df_test['ad_document_id'] = ad_document_id

# joining display_id and user

df_all['uuid'] = df_events.iloc[df_all.display_id - 1].reset_index(drop=1)
df_test['uuid'] = df_events.iloc[df_test.display_id - 1].reset_index(drop=1)


# extracting the leak

def is_leak(doc_id, uuid):
    if doc_id in leak_uuid_dict:
        if uuid in leak_uuid_dict[doc_id]:
            return 1
    return 0

df_all['leak'] = df_all.ad_document_id.combine(df_all.uuid, is_leak)
df_test['leak'] = df_test.ad_document_id.combine(df_test.uuid, is_leak)

df_all['doc_known_views'] = df_all.ad_document_id.apply(lambda d: docs_size.get(d, 0))
df_test['doc_known_views'] = df_test.ad_document_id.apply(lambda d: docs_size.get(d, 0))

df_train_0 = df_all[df_all.fold == 0]
df_train_1 = df_all[df_all.fold == 1]

np.save('features/leak_0.npy', df_train_0.leak.values)
np.save('features/leak_1.npy', df_train_1.leak.values)
np.save('features/leak_test.npy', df_test.leak.values)

np.save('features/doc_known_views_0.npy', df_train_0.doc_known_views.values)
np.save('features/doc_known_views_1.npy', df_train_1.doc_known_views.values)
np.save('features/doc_known_views_test.npy', df_test.doc_known_views.values)