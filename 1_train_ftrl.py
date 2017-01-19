# use pypy for running this script

import re
from time import time
from csv import DictReader
from time import time

import ftrl
from ml_metrics_auc import auc

spaces = re.compile(r' +')


# model parameters

alpha = 0.1
beta = 0.0
L1 = 2.0
L2 = 0.0

D = 2 ** 25

interactions = True
n_epochs = 1
show_auc = False

models = {}
models['0'] = ftrl.FtrlProximal(alpha, beta, L1, L2, D, interactions)
models['1'] = ftrl.FtrlProximal(alpha, beta, L1, L2, D, interactions)
model_full  = ftrl.FtrlProximal(alpha, beta, L1, L2, D, interactions)


# training the models


t0 = time()

print('trainning models...')

for i in range(n_epochs):
    print('epoch %d...' % i)

    with open('tmp/svm_features_train.csv', 'r') as f:
        reader = DictReader(f)

        cnt = 0
        for row in reader:
            y = int(row['clicked'])

            x = spaces.split(row['ad_display_str'].strip())

            if row['fold'] == '0':
                fold = '1'
            else: # '1'
                fold = '0'

            models[fold].fit(x, y)
            model_full.fit(x, y)

            cnt = cnt + 1
            if cnt % 1000000 == 0:
                print('processed %dth row' % cnt)


print('training took %0.3fm' % ((time() - t0) / 60))


# validation and oof prediction

print('validating models...')

t0 = time()

all_y = {'0': [], '1': []}
all_pred = {'0': [], '1': []}

f_pred = {}
f_pred['0'] = open('predictions/ftrl_pred_0.txt', 'w')
f_pred['0'].write('y_actual,y_pred\n')

f_pred['1'] = open('predictions/ftrl_pred_1.txt', 'w')
f_pred['1'].write('y_actual,y_pred\n')

with open('tmp/svm_features_train.csv', 'r') as f:
    reader = DictReader(f)

    cnt = 0
    for row in reader:
        y = int(row['clicked'])
        fold = row['fold']

        x = spaces.split(row['ad_display_str'].strip())
        y_pred = models[fold].predict(x)

        all_y[fold].append(y)
        all_pred[fold].append(y_pred)
        f_pred[fold].write('%s,%s\n' % (y, y_pred))

        cnt = cnt + 1
        if cnt % 1000000 == 0:
            print('processed %dth row' % cnt)
        if show_auc and cnt % 5000000 == 0:
            auc0 = auc(all_y['0'], all_pred['0'])
            auc1 = auc(all_y['1'], all_pred['1'])
            print('auc: %.4f, %.4f' % (auc0, auc1))

auc0 = auc(all_y['0'], all_pred['0'])
auc1 = auc(all_y['1'], all_pred['1'])            
print('final auc: %.4f, %.4f' % (auc0, auc1))

f_pred['0'].close()
f_pred['1'].close()    

print('predict took %0.3fm' % ((time() - t0) / 60))
del all_y, all_pred


# predicting the results on test

print('applying the model to the test data...')

t0 = time()

f_pred = open('predictions/ftrl_pred_test.txt', 'w')
f_pred.write('y_pred\n')

with open('tmp/svm_features_test.csv', 'r') as f:
    reader = DictReader(f)

    cnt = 0
    for row in reader:
        x = spaces.split(row['ad_display_str'].strip())
        y_pred = model_full.predict(x)
        f_pred.write('%s\n' % y_pred)

        cnt = cnt + 1
        if cnt % 1000000 == 0:
            print('processed %dth row' % cnt)

f_pred.close()

print('predict took %0.3fm' % ((time() - t0) / 60))
