import feather
import numpy as np


df_all = feather.read_dataframe('tmp/clicks_train_50_50.feather')

df_train_0 = df_all[df_all.fold == 0].reset_index(drop=1)
df_train_1 = df_all[df_all.fold == 1].reset_index(drop=1)
del df_train_0['fold'], df_train_1['fold'], df_all


# define subfolds for each fold
np.random.seed(1)

uniq0 = df_train_0.display_id.unique()
uniq1 = df_train_1.display_id.unique()

np.random.shuffle(uniq0)
np.random.shuffle(uniq1)

n0 = len(uniq0) // 2
fold_0_0 = set(uniq0[:n0])

n1 = len(uniq1) // 2
fold_1_0 = set(uniq1[:n1])


df_train_0['subfold'] = df_train_0.display_id.isin(fold_0_0).astype('uint8')
df_train_1['subfold'] = df_train_1.display_id.isin(fold_1_0).astype('uint8')

np.save('tmp/fold_0_split.npy', df_train_0.fold.values)
np.save('tmp/fold_1_split.npy', df_train_1.fold.values)


# split fold 0 into subfolds

f_0 = open('ffm/ffm_xgb_0_0.txt', 'w')
f_1 = open('ffm/ffm_xgb_0_1.txt', 'w')

with open('ffm/ffm_xgb_0.txt', 'r') as f_in:
    for fold, line in tqdm(zip(df_train_0.fold, f_in)):
        if fold == 0:
            f_0.write(line)
        else:
            f_1.write(line)

f_0.close()
f_1.close()


# split fold 1 into subfolds

f_0 = open('ffm/ffm_xgb_1_0.txt', 'w')
f_1 = open('ffm/ffm_xgb_1_1.txt', 'w')

with open('ffm/ffm_xgb_1.txt', 'r') as f_in:
    for fold, line in tqdm(zip(df_train_1.fold, f_in)):
        if fold == 0:
            f_0.write(line)
        else:
            f_1.write(line)

f_0.close()
f_1.close()

