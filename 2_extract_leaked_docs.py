# run it with pypy
# taken from 
# https://www.kaggle.com/jiweiliu/outbrain-click-prediction/extract-leak-in-30-mins-with-small-memory

import csv
import os

leak = {}

with open('../data/promoted_content.csv') as f:
    promoted = csv.DictReader(f)
    for c, row in enumerate(promoted):
        if row['document_id'] != '':
            leak[row['document_id']] = 1 


with open('../data/page_views.csv') as f:
    page_views = csv.DictReader(f)
    for c, row in enumerate(page_views):
        if c % 1000000 == 0:
            print c

        doc_id = row['document_id']

        if doc_id not in leak:
            continue

        if leak[doc_id] == 1:
            leak[doc_id] = set()

        lu = len(leak[doc_id])
        leak[doc_id].add(row['uuid'])


with open('tmp/leaked_docs.csv', 'w') as fo:
    fo.write('document_id,uuids\n')
    for k, v in leak.items():
        if v == 1:
            continue

        fo.write('%s,%s\n' % (k, ' '.join(v)))
