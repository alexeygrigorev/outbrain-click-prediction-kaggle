# coding: utf-8

import os

import pandas as pd
import numpy as np
import xgboost as xgb
import feather
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from itertools import combinations



df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')
df_test = feather.read_dataframe('tmp/clicks_test.feather')


# event features:
# - geo
# - time
# - user
# - platform


df_display = pd.read_csv('../data/events.csv')
df_display.geo_location.fillna('', inplace=1)

# geo features

df_geo = df_display.geo_location.str.split('>', expand=True)
df_geo.fillna('*', inplace=1)
df_geo.columns = ['geo_0', 'geo_1', 'geo_2']
del df_geo['geo_2']
df_geo['geo_second_lev'] = df_geo['geo_0'] + '>' + df_geo['geo_1']
del df_geo['geo_1']

df_display['geo_0'] = df_geo['geo_0']
df_display['geo_1'] = df_geo['geo_second_lev']
df_display.rename(columns={'geo_location': 'geo_2'}, inplace=1)
del df_geo

# time features

ts = (df_display.timestamp + 1465876799998) / 1000 - (4 * 60 * 60)
df_display.timestamp = pd.to_datetime(ts, unit='s')

dt = df_display.timestamp.dt
df_display['day'] = dt.dayofweek.astype('str')
df_display['hour'] = dt.hour.astype('str')

del df_display['timestamp'], dt, ts

# platform

df_display.platform = df_display.platform.astype('str')
del df_display['display_id']


# user: convert to base 32 to occupy less space

df_display['user_id'] = LabelEncoder().fit_transform(df_display.uuid)
del df_display['uuid']

def base32(i):
    return np.base_repr(i, base=32)

df_display['user_id'] = df_display['user_id'].apply(base32)



# document features:
# - top category
# - top entity
# - top topic
# - meta: publisher, source

df_ads = pd.read_csv('../data/promoted_content.csv')
ad_to_idx = dict(zip(df_ads.ad_id, df_ads.index))

ads_docs = set(df_display.document_id)
ads_docs.update(df_ads.document_id)


# document categories

df_doc_cat = pd.read_csv('../data/documents_categories.csv')

df_doc_cat = df_doc_cat.drop_duplicates(subset='document_id', keep='first')
df_doc_cat = df_doc_cat[df_doc_cat.confidence_level >= 0.8]
df_doc_cat = df_doc_cat[df_doc_cat.document_id.isin(ads_docs)]

cat_counts = df_doc_cat.category_id.value_counts()
freq_cats = set(cat_counts[cat_counts >= 5].index)

df_doc_cat = df_doc_cat[df_doc_cat.category_id.isin(freq_cats)]

doc_top_cat = dict(zip(df_doc_cat.document_id, df_doc_cat.category_id))
del freq_cats, cat_counts, df_doc_cat


# document entities: hash them to occupy less space

D = 2 ** 24
def entity_name_reduce(entity):
    return '%x' % abs(hash(entity) % D)


df_doc_entities = pd.read_csv('../data/documents_entities.csv')

df_doc_entities = df_doc_entities[df_doc_entities.confidence_level >= 0.8]
df_doc_entities = df_doc_entities[df_doc_entities.document_id.isin(ads_docs)]

df_doc_entities = df_doc_entities.drop_duplicates(subset='document_id', keep='first')
df_doc_entities = df_doc_entities.reset_index(drop=1)

df_doc_entities.entity_id = df_doc_entities.entity_id.apply(entity_name_reduce)

entity_counts = df_doc_entities.entity_id.value_counts()
freq_entites = set(entity_counts[entity_counts >= 5].index)
df_doc_entities = df_doc_entities[df_doc_entities.entity_id.isin(freq_entites)]

doc_top_entity = dict(zip(df_doc_entities.document_id, df_doc_entities.entity_id))

del df_doc_entities, entity_counts, freq_entites


# document topics

df_doc_topics = pd.read_csv('../data/documents_topics.csv')

df_doc_topics = df_doc_topics[df_doc_topics.confidence_level >= 0.8]
df_doc_topics = df_doc_topics[df_doc_topics.document_id.isin(ads_docs)]

df_doc_topics = df_doc_topics.drop_duplicates(subset='document_id', keep='first')
df_doc_topics = df_doc_topics.reset_index(drop=1)

topic_cnt = df_doc_topics.topic_id.value_counts()
freq_topics = set(topic_cnt[topic_cnt >= 5].index)

df_doc_topics = df_doc_topics[df_doc_topics.topic_id.isin(freq_topics)]
doc_top_topic = dict(zip(df_doc_topics.document_id, df_doc_topics.topic_id))

del df_doc_topics, topic_cnt, freq_topics


# document meta info

df_doc_meta = pd.read_csv('../data/documents_meta.csv')
df_doc_meta = df_doc_meta[df_doc_meta.document_id.isin(ads_docs)]
del df_doc_meta['publish_time']

df_doc_meta.source_id.fillna(0, inplace=1)
df_doc_meta.source_id = df_doc_meta.source_id.astype('uint32')

df_doc_meta.publisher_id.fillna(0, inplace=1)
df_doc_meta.publisher_id = df_doc_meta.publisher_id.astype('uint32')

df_doc_meta = df_doc_meta.reset_index(drop=1)
meta_idx = dict(zip(df_doc_meta.document_id, df_doc_meta.index))



# to avoid confusion, let's rename document_id columns

df_display.rename(columns={'document_id': 'on_document_id'}, inplace=1)
df_ads.rename(columns={'document_id': 'ad_document_id'}, inplace=1)


# we will do everything in batches
def prepare_batch(batch):
    batch = batch.reset_index(drop=1)
    
    batch_display = df_display.iloc[batch.display_id - 1].reset_index(drop=1)

    batch_ad_ids = batch.ad_id.apply(ad_to_idx.get)
    batch_ads = df_ads.iloc[batch_ad_ids].reset_index(drop=1)
    del batch_ads['ad_id']

    batch_meta_idx = batch_ads.ad_document_id.apply(meta_idx.get)
    batch_ad_doc_meta = df_doc_meta.iloc[batch_meta_idx].reset_index(drop=1)

    batch_ad_doc_meta['top_entity'] = \
            batch_ad_doc_meta.document_id.apply(lambda did: doc_top_entity.get(did, 'unk'))
    batch_ad_doc_meta['top_topic'] = \
            batch_ad_doc_meta.document_id.apply(lambda did: doc_top_topic.get(did, 'unk'))
    batch_ad_doc_meta['top_cat'] = \
            batch_ad_doc_meta.document_id.apply(lambda did: doc_top_cat.get(did, 'unk'))

    del batch_ad_doc_meta['document_id']

    batch_ad_doc_meta.columns = ['ad_doc_%s' % c for c in batch_ad_doc_meta.columns]

    batch_meta_idx = batch_display.on_document_id.apply(meta_idx.get)
    batch_on_doc_meta = df_doc_meta.iloc[batch_meta_idx].reset_index(drop=1)

    batch_on_doc_meta['top_entity'] = \
            batch_on_doc_meta.document_id.apply(lambda did: doc_top_entity.get(did, 'unk'))
    batch_on_doc_meta['top_topic'] = \
            batch_on_doc_meta.document_id.apply(lambda did: doc_top_topic.get(did, 'unk'))
    batch_on_doc_meta['top_cat'] = \
            batch_on_doc_meta.document_id.apply(lambda did: doc_top_cat.get(did, 'unk'))

    del batch_on_doc_meta['document_id']

    batch_on_doc_meta.columns = ['on_doc_%s' % c for c in batch_on_doc_meta.columns]
    
    joined_batch = pd.concat([batch, batch_ads, batch_display, 
                              batch_ad_doc_meta, batch_on_doc_meta], axis=1)

    for c in ['ad_doc_source_id', 'ad_doc_publisher_id', 'ad_document_id', 'ad_doc_top_cat',
          'on_doc_source_id', 'on_doc_publisher_id', 'on_document_id', 'on_doc_top_cat',
          'ad_id', 'campaign_id', 'advertiser_id']:
        joined_batch[c] = joined_batch[c].astype('str')

    joined_batch.fillna('unk', inplace=1)
    all_features = set(joined_batch.columns) - {'clicked', 'fold', 'display_id'}

    for c in sorted(all_features):
        if 'on_doc' in c or 'geo' in c or c in {'day', 'hour', 'user_id', 'ad_id'}:
            continue

        for c2 in ['day', 'hour', 'geo_0', 'geo_1', 'geo_2']:
            joined_batch['%s_%s' % (c, c2)] = joined_batch[c] + '_' + joined_batch[c2]
    
    two_way_comb = sorted(all_features - {'day', 'hour', 'geo_0', 'geo_1', 'geo_2'})
    
    combs = list(combinations(two_way_comb, 2))

    for c1, c2 in combs:
        if 'on_doc' in c1 and 'on_doc' in c2:
            continue
        joined_batch['%s_%s' % (c1, c2)] = joined_batch[c1].astype('str') + '_' + joined_batch[c2].astype('str')

    return joined_batch



def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
def chunk_dataframe(df, n):
    for i in range(0, len(df), n):
        yield df.iloc[i:i+n]


# apply to train

df = feather.read_dataframe('tmp/clicks_train_50_50.feather')

delete_file_if_exists('tmp/categorical_joined_train.csv')

for batch in tqdm(chunk_dataframe(df, n=100000)):
    batch = prepare_batch(batch)
    append_to_csv(batch, 'tmp/categorical_joined_train.csv')


# apply to test

df = feather.read_dataframe('tmp/clicks_test.feather')

delete_file_if_exists('tmp/categorical_joined_test.csv')

for batch in tqdm(chunk_dataframe(df, n=100000)):
    batch = prepare_batch(batch)
    append_to_csv(batch, 'tmp/categorical_joined_test.csv')
