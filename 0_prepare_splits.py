
import pandas as pd
import numpy as np
import feather

# prepare train split
df_all = pd.read_csv("../data/clicks_train.csv")
df_all.display_id = df_all.display_id.astype('uint32')
df_all.ad_id = df_all.ad_id.astype('uint32')
df_all.clicked = df_all.clicked.astype('uint8')

ids = df_all.display_id.unique()
np.random.seed(1)
np.random.shuffle(ids)

val_size = int(len(ids) * 0.5)
val_display_ids = set(ids[:val_size])

df_all['fold'] = 0

is_val = df_all.display_id.isin(val_display_ids)
df_all.loc[is_val, 'fold'] = 1
df_all.fold = df_all.fold.astype('uint8')

feather.write_dataframe(df_all, 'tmp/clicks_train_50_50.feather')


# prepare test data

df_test = pd.read_csv("../data/clicks_test.csv")
df_test.display_id = df_test.display_id.astype('uint32')
df_test.ad_id = df_test.ad_id.astype('uint32')

feather.write_dataframe(df_test, 'tmp/clicks_test.feather')