#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-01-24
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd
from joblib import load

node2vec = load("data/node2vec.pkl")
model = node2vec.model
df = pd.DataFrame()
df['node_id'] = model.wv.index2word
df['node_id'] = df['node_id'].astype('int32')
n_components=node2vec.n_components
df[list(range(n_components))] = model.wv.vectors
# 拆成用户和商家两个
df['node_id'] = df['node_id'].map({v: k for k, v in load('data/entity2id.pkl').items()})
import numpy as np

is_u = np.array([x[0] == 'u' for x in df['node_id']])
df_u = df.loc[is_u].reset_index(drop=True)
df_m = df.loc[~is_u].reset_index(drop=True)
df_u['node_id'] = df_u['node_id'].apply(lambda x: x[1:]).astype('int32')
df_m['node_id'] = df_m['node_id'].apply(lambda x: x[1:]).astype('int32')
df_u.columns=['user_id']+[f'user_n2v_{i}' for i in range(n_components)]
df_m.columns=['merchant_id']+[f'merchant_n2v_{i}' for i in range(n_components)]
# df_m.rename(columns={'node_id': 'merchant_id'}, inplace=True)
df_u.to_pickle('data/user_n2v.pkl')
df_m.to_pickle('data/merchant_n2v.pkl')
