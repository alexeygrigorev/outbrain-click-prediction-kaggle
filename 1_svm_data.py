# coding: utf-8

import os 
from time import time

import pandas as pd
import numpy as np

from tqdm import tqdm
import feather


# events data

def paths(tokens):
    all_paths = ['_'.join(tokens[0:(i+1)]) for i in range(len(tokens))]
    return ' '.join(all_paths)

def unwrap_geo(geo):
    geo = geo.split('>')
    return paths(geo)


df_events = pd.read_csv("../data/events.csv")
df_events.geo_location.fillna('', inplace=1)

geo_str = df_events.geo_location.apply(unwrap_geo)


ts = (df_events.timestamp + 1465876799998) / 1000
df_events.timestamp = pd.to_datetime(ts, unit='s')


dt = df_events.timestamp.dt

dow = 'dow_' + dt.dayofweek.astype('str')
hours = 'hour_' + dt.hour.astype('str')
dow_hour = 'dow_hour_' + dt.dayofweek.astype('str') + '_' + dt.hour.astype('str')

display_str = 'u_' + df_events.uuid + ' ' + \
              'd_' + df_events.document_id.astype('str') + ' ' + \
              'p_' + df_events.platform.astype('str') + ' ' +  \
              dow + ' ' + hours + ' ' + dow_hour + ' ' + \
              geo_str

df_events_processed = pd.DataFrame()
df_events_processed['display_id'] = df_events.display_id
df_events_processed['display_str'] = display_str


# ad documents data

df_promoted = pd.read_csv("../data/promoted_content.csv")

ad_string = 'addoc_' + df_promoted.document_id.astype('str') + ' ' \
            'campaign_' + df_promoted.campaign_id.astype('str') + ' ' \
            'adv_' + df_promoted.advertiser_id.astype('str') 

df_promoted_processed = pd.DataFrame()
df_promoted_processed['ad_id'] = df_promoted.ad_id
df_promoted_processed['promoted_ad_str'] = ad_string


ad_to_idx = dict(zip(df_promoted_processed.ad_id, df_promoted_processed.index))


# processing data in batches

def prepare_batch(batch):
    batch = batch.reset_index(drop=1)
    
    promoted_idx = batch.ad_id.apply(ad_to_idx.get)
    promoted_ad_str = df_promoted_processed.promoted_ad_str.iloc[promoted_idx]

    display_str = df_events_processed.display_str.iloc[batch.display_id - 1]
    
    promoted_ad_str.reset_index(drop=1, inplace=1)
    display_str.reset_index(drop=1, inplace=1)

    batch['ad_display_str'] = promoted_ad_str + ' ' + display_str
    return batch


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


# preparing data for train & test

df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')

delete_file_if_exists('tmp/svm_features_train.csv')

for batch in tqdm(chunk_dataframe(df_all, n=1000000)):
    batch = prepare_batch(batch)
    append_to_csv(batch, 'tmp/svm_features_train.csv')


df_test = feather.read_dataframe('tmp/clicks_test.feather')

delete_file_if_exists('tmp/svm_features_test.csv')

for batch in tqdm(chunk_dataframe(df_test, n=1000000)):
    batch = prepare_batch(batch)
    append_to_csv(batch, 'tmp/svm_features_test.csv')
