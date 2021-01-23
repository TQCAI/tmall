#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import gensim
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import reduce_mem_usage

user_seq = pd.read_pickle('user_seq.pkl')

size = 100
model = gensim.models.Word2Vec(
    user_seq['user_seq'].apply(lambda x: x.split(' ')),
    size=size,
    window=5,
    min_count=5,
    workers=12
)
user_w2v = pd.DataFrame()
user_w2v['user_id'] = model.wv.index2word
user_w2v[[f'user_w2v_{i}' for i in range(size)]] = model.wv.vectors

merchant_seq = pd.read_pickle('merchant_seq.pkl')
model = gensim.models.Word2Vec(
    merchant_seq['merchant_seq'].apply(lambda x: x.split(' ')),
    size=size,
    window=5,
    min_count=5,
    workers=12
)
merchant_w2v = pd.DataFrame()
merchant_w2v['merchant_id'] = model.wv.index2word
merchant_w2v[[f'merchant_w2v_{i}' for i in range(size)]] = model.wv.vectors

train = pd.read_csv('data_format1/train_format1.csv')
train[['user_id', 'merchant_id']] = train[['user_id', 'merchant_id']].astype('str')
train = train.merge(user_w2v,'left', on='user_id')
train = train.merge(merchant_w2v,'left', on='merchant_id')
train.drop(['user_id', 'merchant_id'], axis=1, inplace=True)
y = train.pop('label')

lr = LogisticRegression()
gbm = LGBMClassifier()
cv = StratifiedKFold(5, True, 0)
score = cross_val_score(gbm, train, y, cv=cv, scoring='roc_auc').mean()
print(score)
merchant_w2v = reduce_mem_usage(merchant_w2v)
user_w2v = reduce_mem_usage(user_w2v)
merchant_w2v['merchant_id'] = merchant_w2v['merchant_id'].astype('int32')
user_w2v['user_id'] = user_w2v['user_id'].astype('int32')
merchant_w2v.to_pickle('merchant_w2v.pkl')
user_w2v.to_pickle('user_w2v.pkl')
