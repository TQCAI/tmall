#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-22
# @Contact    : qichun.tang@bupt.edu.cn
import os

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from joblib import load
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from fesys import FeaturesBuilder


def calc_uv_cosine_mv(df):
    df['uv_cosine_mv'] = np.sum(df[merchant_w2v_col].values * df[user_w2v_col].values, axis=1) / \
                         (np.linalg.norm(df[merchant_w2v_col].values, axis=1) *
                          np.linalg.norm(df[user_w2v_col].values, axis=1))


def apply_boruta_feature_selection(df, boruta):
    id_c = ['user_id', 'merchant_id']
    ids = df[id_c]
    df = boruta.transform(df, weak=True, return_df=True)
    df[id_c] = ids
    return df


def apply_embedding_features(df):
    df = df.merge(user_w2v, 'left', on='user_id')
    df = df.merge(merchant_w2v, 'left', on='merchant_id')
    calc_uv_cosine_mv(df)
    df[sub_col] = df[user_w2v_col].values - df[merchant_w2v_col].values
    df.drop(user_w2v_col + merchant_w2v_col, axis=1, inplace=True)
    return df


boruta = load('data/boruta2.pkl')
similarity_features = pd.read_pickle("data/similarity_features.pkl")
sim_cols = similarity_features.columns.tolist()[2:]
feat_builder: FeaturesBuilder = load('data/feat_builder.pkl')
feat_builder2: FeaturesBuilder = load('data/feat_builder2.pkl')
u_df = feat_builder2.pk2df[('user_id',)]
m_df = feat_builder2.pk2df[('merchant_id',)]
um_df = feat_builder2.pk2df[('user_id', 'merchant_id')]
m_df.drop(['purchase-merchant_id-cnt', 'merchant_id-cnt'], axis=1, inplace=True)
u_df.drop(['purchase-user_id-cnt', 'user_id-cnt'], axis=1, inplace=True)
um_df.drop(['purchase-user_id-merchant_id-cnt', 'user_id-merchant_id-cnt'], axis=1, inplace=True)
for df in [u_df, m_df, um_df]:
    df.drop([c for c in df.columns if '.' in c], axis=1, inplace=True)

train_df = pd.read_csv('data_format1/train_format1.csv')
user_info = pd.read_pickle('data/user_info.pkl')
train_df = train_df.merge(user_info, 'left', on='user_id')
origin_map = feat_builder.pk2df
used_keys = [('user_id',), ('merchant_id',), ('user_id', 'merchant_id'), ('age_range',), ('gender',)]
feat_builder.pk2df = {k: feat_builder.pk2df[k] for k in used_keys}
# train = feat_builder2.outputFeatures(train)
merchant_w2v = pd.read_pickle('data/merchant_n2v.pkl')
user_w2v = pd.read_pickle('data/user_n2v.pkl')
merchant_w2v_col = merchant_w2v.columns.tolist()[1:]
user_w2v_col = user_w2v.columns.tolist()[1:]
sub_col = [f'{c1}-sub-{c2}' for c1, c2 in zip(user_w2v_col, merchant_w2v_col)]
# train_all=feat_builder2.outputFeatures(feat_builder.outputFeatures(train_df))
# dump(train_all, "data/train2.pkl")

y = train_df.pop('label')
N = y.size
# 构造train
train = feat_builder.outputFeatures(train_df)
train = feat_builder2.outputFeatures(train)
# boruta特征筛选
train = apply_boruta_feature_selection(train, boruta)
# 引入Embedding
train = apply_embedding_features(train)
# 引入一些相似度特征 todo: 逐个尝试？
train[sim_cols] = similarity_features.iloc[:N, 2:]

gbm = LGBMClassifier(random_state=0)
cv = StratifiedKFold(5, True, 0)
bc = BalancedBaggingClassifier(
    gbm, random_state=0,
    oob_score=True,  # warm_start=True
)

prediction = pd.read_csv('data_format1/test_format1.csv')
prediction.pop('prob')
test_df = prediction.merge(user_info, 'left', on='user_id')

# 构造test
test = feat_builder.outputFeatures(test_df)
test = feat_builder2.outputFeatures(test)
# boruta特征筛选
test = apply_boruta_feature_selection(test, boruta)
# 引入Embedding
test = apply_embedding_features(test)
# 用户与商家的余弦距离
# 引入一些相似度特征
test[sim_cols] = similarity_features.iloc[N:, 2:]

model = bc.fit(train, y)
y_pred = bc.predict_proba(test)
prediction['prob'] = y_pred[:, 1]
prediction.to_csv('predictions/prediction2.csv', index=False)
os.system('google-chrome https://ssl.gstatic.com/dictionary/static/sounds/oxford/ok--_gb_1.mp3')
print(bc)
